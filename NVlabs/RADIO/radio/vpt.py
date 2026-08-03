import math
from typing import List

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class DeepVPTTuner(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_prompts: int,
        embed_dim: int,
        patch_size: int = 16,
        project_dim: int = -1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_prompts = num_prompts
        self.embed_dim = embed_dim

        self.prompt_dropout = nn.Dropout(dropout)
        if project_dim > -1:
            prompt_dim = project_dim
            self.prompt_proj = nn.Linear(prompt_dim, embed_dim)
            nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode="fan_out")
        else:
            prompt_dim = embed_dim
            self.prompt_proj = nn.Identity()

        val = math.sqrt(6.0 / float(3 * patch_size * patch_size + prompt_dim))
        self.prompts = nn.Parameter(torch.zeros(num_layers, num_prompts, prompt_dim))
        nn.init.uniform_(self.prompts.data, -val, val)


class PromptInjector(nn.Module):
    def __init__(
        self,
        original_blocks: nn.Module,
        num_prompts: int = 10,
        embed_dim: int = 1280,
        num_skip: int = 1,
        insert_start_layer: int = 1,
        insert_end_layer: int = -1,
        patch_size: int = 16,
        project_dim: int = -1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.original_blocks = original_blocks
        self.num_prompts = num_prompts
        self.num_layers = len(original_blocks)
        self.num_skip = num_skip
        self.embed_dim = embed_dim
        self.active_layers = self._build_active_layers(insert_start_layer, insert_end_layer)
        self.layer_to_prompt_idx = {
            layer_idx: prompt_idx for prompt_idx, layer_idx in enumerate(self.active_layers)
        }

        self.is_active = True
        self.grad_checkpointing = False
        self.supports_gradient_checkpointing = True

        self.tuner = DeepVPTTuner(
            len(self.active_layers),
            num_prompts,
            embed_dim,
            patch_size=patch_size,
            project_dim=project_dim,
            dropout=dropout,
        )

    def _build_active_layers(self, insert_start_layer: int, insert_end_layer: int) -> List[int]:
        # Layer indices are 1-based from the input side:
        # layer 1 is the first ViT block that sees image tokens.
        start_idx = 0 if insert_start_layer is None else int(insert_start_layer) - 1
        end_idx = (
            self.num_layers - 1
            if insert_end_layer is None or int(insert_end_layer) < 0
            else int(insert_end_layer) - 1
        )

        start_idx = max(0, start_idx)
        end_idx = min(self.num_layers - 1, end_idx)

        if start_idx > end_idx:
            raise ValueError(
                f"Invalid VPT layer range: start={insert_start_layer}, end={insert_end_layer}, num_layers={self.num_layers}"
            )

        return list(range(start_idx, end_idx + 1))

    def _forward_block(self, x: torch.Tensor, block_idx: int):
        block = self.original_blocks[block_idx]
        prompt_idx = self.layer_to_prompt_idx.get(block_idx)

        if prompt_idx is None:
            return block(x)

        batch_size = x.shape[0]
        raw_prompts = self.tuner.prompts[prompt_idx]
        projected_prompts = self.tuner.prompt_proj(raw_prompts)
        layer_prompts = projected_prompts.unsqueeze(0).expand(batch_size, -1, -1)
        layer_prompts = self.tuner.prompt_dropout(layer_prompts)

        skip_tokens = x[:, :self.num_skip, :]
        patch_tokens = x[:, self.num_skip :, :]
        x_in = torch.cat([skip_tokens, layer_prompts, patch_tokens], dim=1)

        x_out = block(x_in)

        skip_tokens_out = x_out[:, :self.num_skip, :]
        patch_tokens_out = x_out[:, self.num_skip + self.num_prompts :, :]
        return torch.cat([skip_tokens_out, patch_tokens_out], dim=1)

    def forward(self, x: torch.Tensor):
        if not self.is_active:
            for block in self.original_blocks:
                x = block(x)
            return x

        use_grad_ckpt = self.training and self.grad_checkpointing and x.requires_grad

        for i in range(self.num_layers):
            if use_grad_ckpt:
                x = checkpoint(
                    lambda hidden_states, idx=i: self._forward_block(hidden_states, idx),
                    x,
                    use_reentrant=False,
                )
            else:
                x = self._forward_block(x, i)

        return x
