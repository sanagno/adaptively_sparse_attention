"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass
from typing import List


import time
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

from utils.root_finding import entmax_bisect
from utils.memory_allocation import DynamicTensorFast, DynamicTensorReferenceDynamic


# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return (
        0.5
        * x
        * (
            1.0
            + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        )
    )


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, block_num, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        self.sparse_attention = config.sparse_attention
        self.block_num = block_num
        self.config = config

        if self.sparse_attention:
            # initialize alpha to 1, this will be overwritten
            # setting alpa to 1 is unstable, so we set it to 1 + eps
            self.sparsity_alpha = 1.000001

            self.int_n_embd = config.int_n_embd if config.int_n_embd else config.n_embd

            self.q_int = nn.Linear(config.n_embd, self.int_n_embd, bias=False)
            self.k_int = nn.Linear(config.n_embd, self.int_n_embd, bias=False)

            self.int_bias = nn.Parameter(
                torch.ones(
                    1,
                )
                * config.sparse_attention_int_bias,
            )

            torch.nn.init.normal_(
                self.q_int.weight, mean=0.0, std=1 / math.sqrt(config.n_embd)
            )
            torch.nn.init.normal_(
                self.k_int.weight, mean=0.0, std=1 / math.sqrt(config.n_embd)
            )

            # bias for the dropping probabilities. Here we assume a token does not drop itself.
            self.register_buffer(
                "bias_int",
                torch.tril(
                    torch.ones(config.block_size, config.block_size), diagonal=-1
                ).view(1, 1, config.block_size, config.block_size),
            )

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # We use torch 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        assert self.flash

        # bias for the attention mask for casual decoding
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(
        self,
        x,
        prev_attn_mask=None,
        mask=None,
        store=None,
        validity_map=None,
        first_generation=False,
    ):
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        device = x.device

        q, k, v = self.c_attn.forward(x).split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        if self.sparse_attention:
            q_int = self.q_int(x)
            k_int = self.k_int(x)

        store_mask = None
        insertion_indices = None

        if store is not None:
            # we are using caching while generating
            if first_generation:
                # if this is the first generation step, we need to insert everything into the store
                for i in range(T):
                    # add everything to the cache
                    if not self.sparse_attention:
                        store.append([k[:, :, i, :], v[:, :, i, :]])
                    else:
                        store.append([k[:, :, i, :], v[:, :, i, :], k_int[:, i, :]])

                # get the correct last insertion indices, due to padding within the prefixes
                insertion_indices = torch.sum(validity_map, dim=-1) - 1

                # After inseting in the store, remove based on the padding mask
                _, store_mask = store.values()
                store_mask = store_mask.clone()
                store_mask[:, : validity_map.shape[-1]] = validity_map

                store.remove(torch.logical_not(store_mask))
            else:
                # add new elements to the store
                if not self.sparse_attention:
                    store.append([k[:, :, 0, :], v[:, :, 0, :]])

                    (k, v), store_mask = store.values()
                else:
                    insertion_indices = store.append(
                        [k[:, :, 0, :], v[:, :, 0, :], k_int[:, 0, :]]
                    )

                    (k, v, k_int), store_mask = store.values()

                validity_map = store_mask

        context_T = k.shape[2]

        if not self.sparse_attention:
            # regular causal attention
            attn_mask = torch.zeros(B, 1, T, context_T, device=x.device, dtype=x.dtype)

            if validity_map is not None:
                # filter out the attention mask to only include the tokens that are not yet processed
                attn_mask = attn_mask.masked_fill(
                    validity_map[:, None, None, :] == 0,
                    float("-inf"),
                )

            cumprobs = 0  # for compatibility
        else:
            p_int_raw = (
                (
                    torch.matmul(q_int, k_int.transpose(-1, -2))
                    / math.sqrt(self.int_n_embd)
                    + self.int_bias
                )
                .unsqueeze(1)
                .unsqueeze(-1)
            )

            if self.sparsity_alpha == "inf":
                # in eval mode we replace the alpha-sigmoid with the step function
                p_int = (p_int_raw > 0)[..., 0]
            else:
                # Compare the raw drop scores with the values 0 to get the drop probabilities.
                p_int_raw = torch.cat([p_int_raw, torch.zeros_like(p_int_raw)], dim=-1)

                # Take only the first value of the entmax_bisect output, which is the probability of dropping.
                p_int = entmax_bisect(p_int_raw.to(torch.float32), self.sparsity_alpha)[
                    ..., 0
                ]

            if store is not None:
                # here we need to drop from the store
                if first_generation:
                    p_int = p_int.float()
                    p_int = p_int.masked_fill(self.bias_int[:, :, :T, :T] == 0, 1)

                    p_int = p_int.masked_fill(validity_map[:, None, None, :] == 0, 0)

                    # Multiply together probs from the previous tokens.
                    cumprobs = torch.cumprod(p_int, dim=-2)

                    attn_mask = torch.log(cumprobs)

                    if prev_attn_mask is not None:
                        attn_mask = attn_mask + prev_attn_mask

                    store_mask[:, : validity_map.shape[-1]] = cumprobs[
                        torch.arange(B, device=device), 0, insertion_indices, :
                    ].bool()

                    store.remove(torch.logical_not(store_mask))
                else:
                    # specify that we cannot drop ourselves
                    p_int[
                        torch.arange(B, device=device), 0, 0, insertion_indices
                    ] = True

                    p_int = torch.logical_and(p_int, validity_map[:, None, None, :])

                    attn_mask = p_int  # scaled_dot_product_attention can also handle boolean masks
                    cumprobs = None

                    store.remove(torch.logical_not(p_int[:, 0, 0, :]))
            else:
                # training phase
                p_int = p_int.masked_fill(self.bias_int[:, :, :T, :T] == 0, 1)

                if validity_map is not None:
                    p_int = p_int.masked_fill(validity_map[:, None, None, :] == 0, 0)

                # Multiply together probs from the previous tokens.
                cumprobs = torch.cumprod(p_int, dim=-2)

                # Just for stability reasons add an epsilon ...
                attn_mask = torch.log(cumprobs + (1e-40 if self.training else 0)).to(
                    p_int_raw.dtype
                )

                if prev_attn_mask is not None:
                    attn_mask = attn_mask + prev_attn_mask

        if T == context_T:
            # Add casual masking, only during training
            attn_mask = attn_mask.masked_fill(
                self.bias[:, :, :T, :T] == 0, float("-inf")
            )

        if mask is not None:  # masking of tokens during training
            attn_mask = attn_mask.masked_fill(
                mask[:, None, None, :] == 0, float("-inf")
            )

        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout
        )
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))

        return y, cumprobs, attn_mask


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, block_num, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(block_num, config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        self.block_num = block_num

    def forward(
        self,
        x,
        attn_mask,
        **kwargs,
    ):
        y, cumprobs, attn_mask = self.attn.forward(
            self.ln_1(x),
            attn_mask,
            **kwargs,
        )
        x = x + y

        x = self.mlp.forward(self.ln_2(x)) + x

        return x, cumprobs, attn_mask


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    int_n_embd: int = None
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    sparse_attention: bool = False
    sparse_attention_int_bias: int = 5
    lm_head: bool = True
    propagate_in_depth: bool = True


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(cnt, config) for cnt in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )

        if config.lm_head:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

            # with weight tying when using torch.compile() some warnings get generated:
            # "UserWarning: functional_call was passed multiple values for tied weights.
            # This behavior is deprecated and will be an error in future versions"
            # not 100% sure what this is, so far seems to be harmless. TODO investigate
            self.transformer.wte.weight = (
                self.lm_head.weight
            )  # https://paperswithcode.com/method/weight-tying
        else:
            self.lm_head = None

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        self.sparse_attention = config.sparse_attention

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= sum([p.numel() for p in self.transformer.wpe.parameters()])
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx,
        targets,
        mask=None,
        return_attn_masks=True,
        return_intermediate_acts=False,
        return_raw=False,
    ):
        device = idx.device
        _, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(
            0
        )  # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(
            pos
        )  # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        if return_attn_masks:
            all_attn_masks = []

        if return_intermediate_acts:
            all_acts = [x.detach().cpu()]

        attn_mask = None
        all_cumprobs = []

        for block in self.transformer.h:
            (x, cumprobs, attn_mask) = block(
                x,
                attn_mask if self.config.propagate_in_depth else None,
                mask=mask,
            )

            all_cumprobs.append(cumprobs)

            if return_attn_masks:
                all_attn_masks.append(attn_mask if attn_mask is not None else None)

            if return_intermediate_acts:
                all_acts.append(x.detach().cpu())

        x = self.transformer.ln_f(x)

        if return_raw:
            return self.lm_head(x), targets, all_attn_masks

        if self.lm_head:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )

        if return_intermediate_acts:
            return logits, all_cumprobs, loss, all_attn_masks, all_acts

        if return_attn_masks:
            return logits, all_cumprobs, loss, all_attn_masks

        return (
            logits,
            all_cumprobs,
            loss,
            None,
        )

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size]
        )
        for block in self.transformer.h:
            block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None, cache_dir=None):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        override_args = override_args or {}  # default to empty dict
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        config_args["bias"] = True  # always True for GPT model checkpoints

        # we can override the dropout rate, if desired
        for k, v in override_args.items():
            assert k in [
                "dropout",
                "sparse_attention",
                "sparse_attention_int_bias",
                "int_n_embd",
                "propagate_in_depth",
            ], f"unable to override arg: {k}"
            print("overriding arg: %s = %s" % (k, v))
            config_args[k] = v

        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        print("cache_dir:", cache_dir)
        model_hf = GPT2LMHeadModel.from_pretrained(model_type, cache_dir=cache_dir)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        # assert len(sd_keys_hf) == len(
        #     sd_keys
        # ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert (
                    sd_hf[k].shape == sd[k].shape
                ), f"{k}: {sd_hf[k].shape} != {sd[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        decay.remove("lm_head.weight")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
        use_fused = (device_type == "cuda") and (
            "fused" in inspect.signature(torch.optim.AdamW).parameters
        )
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_length,
        validity_mask=None,  # used for unequal length prefixes
        temperature=1.0,
        do_sample=True,
        top_k=None,
        eos_token_id=None,
        measure_time=False,
    ):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """

        B, _ = idx.shape
        device = idx.device
        batch_arange = torch.arange(B).to(device)

        time_per_token = []
        sizes_per_store = []
        load_factors = []

        if validity_mask is None:
            validity_mask = torch.ones_like(idx)

        current_length = validity_mask.sum(dim=1)
        initial_capacity = 64

        # store for key-values, one per layer
        if self.sparse_attention:
            # Dynamic storage tha allows for fast insertion and removal of elements
            # we need to store key values and the interaction keys in this case
            Storage = [
                DynamicTensorFast(
                    B,
                    [
                        (self.config.n_head, self.config.n_embd // self.config.n_head),
                        (self.config.n_head, self.config.n_embd // self.config.n_head),
                        self.config.int_n_embd,
                    ],
                    initial_capacity,
                    reduce_fragmentation=False,
                    device=device,
                    debug=False,
                )
                for _ in range(self.config.n_layer)
            ]
        else:
            # if we are not using sparse attention, we can use the more efficient DynamicTensorReferenceDynamic
            Storage = [
                DynamicTensorReferenceDynamic(
                    B,
                    [
                        (self.config.n_head, self.config.n_embd // self.config.n_head),
                        (self.config.n_head, self.config.n_embd // self.config.n_head),
                    ],
                    initial_capacity,
                    reduce_fragmentation=False,
                    device=device,
                    debug=False,
                )
                for _ in range(self.config.n_layer)
            ]

        first_generation = True

        while True:
            if idx.shape[1] >= max_length or (
                eos_token_id is not None and (idx[:, -1] == eos_token_id).all()
            ):
                break

            if measure_time:
                if "cuda" in device.type:
                    torch.cuda.synchronize()
                start = time.time()

            if first_generation:
                current_idx = idx
                pos = torch.arange(
                    0, idx.shape[1], dtype=torch.long, device=device
                ).unsqueeze(0)
            else:
                current_idx = idx[batch_arange, current_length - 1].unsqueeze(1)
                pos = (current_length - 1).unsqueeze(1)

            tok_emb = self.transformer.wte(current_idx)
            pos_emb = self.transformer.wpe(pos)
            x = self.transformer.drop(tok_emb + pos_emb)

            attn_mask = None
            for block_idx, block in enumerate(self.transformer.h):
                (x, _, attn_mask) = block(
                    x,
                    attn_mask if self.config.propagate_in_depth else None,
                    validity_map=validity_mask if first_generation else None,
                    store=Storage[block_idx],
                    first_generation=first_generation,
                )

            x = self.transformer.ln_f(x)

            if not first_generation:
                # x already should have a length of just 1
                logits = self.lm_head(x[:, -1, :])
            else:
                logits = self.lm_head(x[batch_arange, current_length - 1, :])

            # pluck the logits at the final step and scale by desired temperature
            logits = logits / temperature

            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)

            if measure_time:
                if "cuda" in device.type:
                    torch.cuda.synchronize()
                time_per_token.append(time.time() - start)

            if do_sample:
                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                # take the most likely option
                _, idx_next = torch.topk(probs, k=1, dim=-1)

            idx = torch.cat(
                [idx, torch.zeros(B, 1, dtype=torch.long, device=device)], dim=1
            )
            idx.scatter_(1, current_length.unsqueeze(1), idx_next)

            first_generation = False
            current_length = current_length + 1

            if self.sparse_attention:
                sizes_per_store.append(
                    torch.tensor([store.max_padded_length for store in Storage])
                )
                load_factors.append(
                    torch.tensor(
                        [
                            torch.sum(store.mask[:, : store.max_padded_length])
                            / store.mask[:, : store.max_padded_length].numel()
                            for store in Storage
                        ]
                    )
                )

        return (
            [idx[i, : current_length[i]] for i in range(idx.shape[0])],
            np.array(time_per_token),
            torch.stack(sizes_per_store, dim=0) if self.sparse_attention else None,
            torch.stack(load_factors, dim=0) if self.sparse_attention else None,
        )
