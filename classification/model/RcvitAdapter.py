    # --------------------------------------------------------
# References:
# https://github.com/jxhe/unify-parameter-efficient-tuning
# --------------------------------------------------------

import torch.nn as nn
import torch
import numpy as np
import math

from torch.cuda.amp import autocast

from einops import rearrange, repeat
import itertools
import os
import copy

from timm.models.layers import DropPath, trunc_normal_, to_2tuple
from timm.models.registry import register_model

import utils as utils

from model import rcvit

# --------------------------------------------------------
class RCvitAdapter(rcvit.RCViT):
    def __init__(self, layers, embed_dims, mlp_ratios=4, downsamples=[True, True, True, True], norm_layer=nn.BatchNorm2d, attn_bias=False,
                 act_layer=nn.GELU, num_classes=1000, drop_rate=0., drop_path_rate=0., fork_feat=False,
                 init_cfg=None, pretrained=None, distillation=True,input_size=224 , tuning_config= None,finetune=None , **kwargs):
        super().__init__(layers, embed_dims, mlp_ratios, downsamples, norm_layer, attn_bias,
                 act_layer, num_classes, drop_rate, drop_path_rate, fork_feat,
                 init_cfg, pretrained, distillation, **kwargs)

        self.config = tuning_config
        self._device = tuning_config._device
        self.emb_size = []
        self.cur_adapter = nn.ModuleList()
        self.input_size = input_size
        self.get_embedding_size()
        self.get_new_adapter()
        self.dist = distillation
        self.freeze()
        self.head = nn.Linear(in_features=220, out_features=num_classes, bias=True)
        self.head.requires_grad = True



        if finetune is not None:
            checkpoint = torch.load(finetune, map_location="cpu")
            state_dict = checkpoint["model"]

            # Remove as chaves da camada `head` para evitar incompatibilidades
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head")}

            # Carrega o state_dict modificado
            self.load_state_dict(state_dict, strict=False)
      
        # Parâmetros treináveis e não treináveis em cada bloco
        block_params = {}
        for i, block in enumerate(self.network):
            trainable = sum(p.numel() for p in block.parameters() if p.requires_grad)
            non_trainable = sum(p.numel() for p in block.parameters() if not p.requires_grad)
            block_params[f"Bloco {i+1}"] = {"treináveis": trainable, "não treináveis": non_trainable}

        print("Parâmetros por bloco:")
        for name, counts in block_params.items():
            print(f"{name}: {counts['treináveis']} treináveis, {counts['não treináveis']} não treináveis")

        # Parâmetros treináveis e não treináveis em cada adaptador
        adapter_params = {}
        for i, adapter in enumerate(self.cur_adapter):
            trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
            non_trainable = sum(p.numel() for p in adapter.parameters() if not p.requires_grad)
            adapter_params[f"Adapter {i+1}"] = {"treináveis": trainable, "não treináveis": non_trainable}

        print("\nParâmetros por adaptador:")
        for name, counts in adapter_params.items():
            print(f"{name}: {counts['treináveis']} treináveis, {counts['não treináveis']} não treináveis")
        
        # Parâmetros da camada de cabeça (head)
        head_trainable = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        head_non_trainable = sum(p.numel() for p in self.head.parameters() if not p.requires_grad)
        print(f"\nParâmetros da cabeça: {head_trainable} treináveis, {head_non_trainable} não treináveis")

        # Parâmetros totais da rede
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_non_trainable = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print(f"\nTotal de parâmetros na rede: {total_trainable} treináveis, {total_non_trainable} não treináveis")

    def freeze(self):
        # Congela todos os parâmetros da rede principal
        for name, param in self.named_parameters():
            if "cur_adapter" not in name:  # Verifica se o parâmetro não pertence aos adapters
                param.requires_grad = False

        # Libera apenas os parâmetros dos adapters para treinamento
        for adapter in self.cur_adapter:
            for param in adapter.parameters():
                param.requires_grad = True

        
    def get_embedding_size(self):
        out = torch.rand((1,3,self.input_size,self.input_size))
        with torch.no_grad():
            out = self.patch_embed(out)
            for idx,block in enumerate(self.network):
                out = block(out)
                self.emb_size.append(out.size(3))

    def get_new_adapter(self):
        config = self.config
        self.cur_adapter = nn.ModuleList()
        if config.ffn_adapt:
            for i in range(len(self.network)):
                self.config.d_model = self.emb_size[i]
                config.fnn_num = self.emb_size[i]/2
                adapter = Adapter(self.config, dropout=0.1, bottleneck=config.ffn_num,
                                        init_option=config.ffn_adapter_init_option,
                                        adapter_scalar=config.ffn_adapter_scalar,
                                        adapter_layernorm_option=config.ffn_adapter_layernorm_option,
                                        ).to(self._device)
                self.cur_adapter.append(adapter)
            self.cur_adapter.requires_grad_(True)
        else:
            print("====Not use adapter===")

    def forward_tokens(self, x):
        for idx, block in enumerate(self.network):
            x = block(x) 
            adapt = self.cur_adapter[idx]
            if adapt is not None:
                adapt_x = adapt(x, add_residual=False)
            else:
                adapt_x = None        
            if adapt_x is not None:
                if self.config.ffn_adapt:
                    if self.config.ffn_option == 'sequential':
                        x = adapt(x)
                    elif self.config.ffn_option == 'parallel':
                        x = x + adapt_x
                    else:
                        raise ValueError(self.config.ffn_adapt)

            
        return x

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.forward_tokens(x)
        if self.fork_feat:
            # otuput features of four stages for dense prediction
            return x
        x = self.norm(x)
        if self.dist:
            cls_out = self.head(x.flatten(2).mean(-1)), self.dist_head(x.flatten(2).mean(-1))
            if not self.training:
                cls_out = (cls_out[0] + cls_out[1]) / 2
        else:
            cls_out = self.head(x.flatten(2).mean(-1))
        # for image classification
        return cls_out

class Adapter(nn.Module):
    def __init__(self,
                 config=None,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="bert",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.n_embd = config.d_model if d_model is None else d_model
        #self.down_size = config.fnn_num if bottleneck is None else bottleneck
        self.down_size = self.n_embd//2
        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output

