# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import argparse
import gc
import math
import os
import random
import re
import types

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file

from data.data_utils import add_special_tokens, pil_img2rgb
from data.transforms import ImageTransform
from inferencer import InterleaveInferencer
from modeling.autoencoder import load_ae
from modeling.bagel import (
    Bagel,
    BagelConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer


DEFAULT_IMAGE = "assets/edit_input.jpg"
DEFAULT_OUTPUT = "assets/edited_image.png"
DEFAULT_NUM_GPUS = 4
DEFAULT_TP_PLAN = "auto"
TP_PLAN = {
    "language_model.model.layers.*.self_attn.q_proj": "colwise",
    "language_model.model.layers.*.self_attn.k_proj": "colwise",
    "language_model.model.layers.*.self_attn.v_proj": "colwise",
    "language_model.model.layers.*.self_attn.q_proj_moe_gen": "colwise",
    "language_model.model.layers.*.self_attn.k_proj_moe_gen": "colwise",
    "language_model.model.layers.*.self_attn.v_proj_moe_gen": "colwise",
    "language_model.model.layers.*.self_attn.o_proj": "rowwise",
    "language_model.model.layers.*.self_attn.o_proj_moe_gen": "rowwise",
    "language_model.model.layers.*.mlp.gate_proj": "colwise",
    "language_model.model.layers.*.mlp.up_proj": "colwise",
    "language_model.model.layers.*.mlp.down_proj": "rowwise",
    "language_model.model.layers.*.mlp_moe_gen.gate_proj": "colwise",
    "language_model.model.layers.*.mlp_moe_gen.up_proj": "colwise",
    "language_model.model.layers.*.mlp_moe_gen.down_proj": "rowwise",
    "language_model.lm_head": "colwise_gather_output",
}
DEFAULT_PROMPT = (
    "Turn the scene into a cozy evening cafe photo: add warm ambient lighting, "
    "soft shadows, and a small vase of fresh flowers on the table while keeping "
    "the main subject and camera angle unchanged."
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run BAGEL image editing inference.")
    parser.add_argument("--model-path", default="models/BAGEL-7B-MoT")
    parser.add_argument("--input-image", default=DEFAULT_IMAGE)
    parser.add_argument("--output-image", default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-timesteps", type=int, default=50)
    parser.add_argument("--cfg-text-scale", type=float, default=4.0)
    parser.add_argument("--cfg-img-scale", type=float, default=2.0)
    parser.add_argument("--cfg-renorm-type", default="text_channel")
    parser.add_argument("--cfg-renorm-min", type=float, default=0.0)
    parser.add_argument("--timestep-shift", type=float, default=3.0)
    parser.add_argument("--num-gpus", type=int, default=DEFAULT_NUM_GPUS)
    parser.add_argument("--tp-plan", default=DEFAULT_TP_PLAN, choices=["auto"])
    parser.add_argument("--allow-non-h100", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def rank_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def init_tensor_parallel(num_gpus, allow_non_h100):
    if "WORLD_SIZE" not in os.environ:
        raise RuntimeError(
            "Tensor parallel inference must be launched with torchrun, for example:\n"
            "torchrun --nproc-per-node 4 image_edit_infer.py --model-path models/BAGEL-7B-MoT"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for 4-GPU BAGEL inference.")

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if world_size != num_gpus:
        raise RuntimeError(f"Expected torchrun world size {num_gpus}, found {world_size}.")
    if torch.cuda.device_count() < num_gpus:
        raise RuntimeError(f"Expected at least {num_gpus} visible CUDA GPUs, found {torch.cuda.device_count()}.")

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    gpu_name = torch.cuda.get_device_name(local_rank)
    if not allow_non_h100 and "H100" not in gpu_name:
        raise RuntimeError(f"Expected rank {rank} local GPU to be an H100, found {gpu_name!r}.")

    return rank, local_rank, world_size, device


def wildcard_module_name(name):
    return re.sub(r"\.\d+(\.|$)", lambda match: ".*" + match.group(1), name)


def get_tp_style(name):
    generic_name = wildcard_module_name(name)
    if generic_name in TP_PLAN:
        return TP_PLAN[generic_name]
    if "." in generic_name:
        parent_name = generic_name.rsplit(".", 1)[0]
        if parent_name in TP_PLAN:
            return TP_PLAN[parent_name]
    return None


def shard_along_dim(tensor, dim, rank, world_size, name):
    if tensor.shape[dim] % world_size != 0:
        raise ValueError(f"Cannot tensor-parallel shard {name}: shape {tuple(tensor.shape)} is not divisible by {world_size}.")
    shard_size = tensor.shape[dim] // world_size
    slices = [slice(None)] * tensor.ndim
    slices[dim] = slice(rank * shard_size, (rank + 1) * shard_size)
    return tensor[tuple(slices)].contiguous()


def shard_for_tp(name, tensor, style, rank, world_size):
    if style in {"colwise", "colwise_gather_output"}:
        return shard_along_dim(tensor, 0, rank, world_size, name)
    if style == "rowwise" and tensor.ndim > 1:
        return shard_along_dim(tensor, 1, rank, world_size, name)
    return tensor.contiguous()


def set_module_tensor(model, name, tensor, requires_grad=None):
    module_name, tensor_name = name.rsplit(".", 1)
    module = model.get_submodule(module_name)
    current = getattr(module, tensor_name)
    if isinstance(current, torch.nn.Parameter):
        if requires_grad is None:
            requires_grad = current.requires_grad
        setattr(module, tensor_name, torch.nn.Parameter(tensor, requires_grad=requires_grad))
    else:
        setattr(module, tensor_name, tensor)


def load_tp_state_dict(model, checkpoint_path, rank, world_size, dtype):
    state_dict = load_file(checkpoint_path, device="cpu")
    model_param_names = set()
    loaded_names = set()
    missing_names = []

    for name, param in model.named_parameters():
        model_param_names.add(name)
        tensor = state_dict.get(name)
        if tensor is None:
            missing_names.append(name)
            continue

        style = get_tp_style(name)
        if style is not None:
            tensor = shard_for_tp(name, tensor, style, rank, world_size)
        if tensor.is_floating_point():
            tensor = tensor.to(dtype=dtype)
        set_module_tensor(model, name, tensor, requires_grad=param.requires_grad)
        loaded_names.add(name)

    for name, buffer in model.named_buffers():
        model_param_names.add(name)
        tensor = state_dict.get(name)
        if tensor is None:
            continue
        if tensor.is_floating_point():
            tensor = tensor.to(dtype=dtype)
        set_module_tensor(model, name, tensor)
        loaded_names.add(name)

    unexpected_names = sorted(set(state_dict.keys()) - loaded_names)
    del state_dict
    gc.collect()
    return missing_names, unexpected_names


def tp_linear_forward(self, input_tensor):
    if self._tp_style == "rowwise":
        output = F.linear(input_tensor, self.weight, None)
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=self._tp_group)
        if self.bias is not None:
            output = output + self.bias
        return output

    output = F.linear(input_tensor, self.weight, self.bias)
    if self._tp_style == "colwise_gather_output":
        chunks = [torch.empty_like(output) for _ in range(self._tp_world_size)]
        dist.all_gather(chunks, output.contiguous(), group=self._tp_group)
        output = torch.cat(chunks, dim=-1).contiguous()
    return output


def attach_tp_to_linear(module, style, world_size):
    module._tp_style = style
    module._tp_world_size = world_size
    module._tp_group = dist.group.WORLD
    module.forward = types.MethodType(tp_linear_forward, module)
    if style == "rowwise" and hasattr(module, "in_features"):
        module.in_features = math.ceil(module.in_features / world_size)
    elif style == "colwise" and hasattr(module, "out_features"):
        module.out_features = math.ceil(module.out_features / world_size)


def adjust_attention_for_tp(model, world_size):
    for module in model.modules():
        attrs = ("num_heads", "num_key_value_heads", "num_key_value_groups", "head_dim", "hidden_size")
        if not all(hasattr(module, attr) for attr in attrs):
            continue
        if getattr(module, "tp_enabled", False):
            continue
        if module.num_heads % world_size != 0:
            raise ValueError(f"Attention heads ({module.num_heads}) must be divisible by TP size {world_size}.")
        if module.num_key_value_heads % world_size != 0:
            raise ValueError(
                f"KV heads ({module.num_key_value_heads}) must be divisible by TP size {world_size}."
            )

        module.tp_enabled = True
        module.tp_full_hidden_size = module.hidden_size
        module.tp_full_num_heads = module.num_heads
        module.tp_full_num_key_value_heads = module.num_key_value_heads
        module.num_heads = module.num_heads // world_size
        module.num_key_value_heads = module.num_key_value_heads // world_size
        module.num_key_value_groups = module.num_heads // module.num_key_value_heads
        module.hidden_size = module.num_heads * module.head_dim


def apply_tensor_parallel(model, world_size):
    tp_modules = 0
    for name, module in model.named_modules():
        style = get_tp_style(name)
        if style is None:
            continue
        if not isinstance(module, torch.nn.Linear):
            raise TypeError(f"TP plan entry {name} matched {type(module).__name__}, expected torch.nn.Linear.")
        attach_tp_to_linear(module, style, world_size)
        tp_modules += 1

    adjust_attention_for_tp(model, world_size)
    model._tp_plan = TP_PLAN
    model._tp_size = world_size
    return tp_modules


def load_inferencer(model_path, device, rank, world_size, tp_plan):
    if tp_plan != "auto":
        raise ValueError("Only tp_plan='auto' is supported.")

    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1

    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
    )

    language_model = Qwen2ForCausalLM(llm_config)
    vit_model = SiglipVisionModel(vit_config)
    model = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    vae_transform = ImageTransform(448, 448, 16)
    vit_transform = ImageTransform(512, 512, 14)

    checkpoint_path = os.path.join(model_path, "ema.safetensors")
    missing, unexpected = load_tp_state_dict(model, checkpoint_path, rank, world_size, torch.bfloat16)
    tp_module_count = apply_tensor_parallel(model, world_size)
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    vae_model = vae_model.to(device).eval()
    torch.cuda.empty_cache()

    rank_print(f"Loaded BAGEL with tp_plan={tp_plan!r} using tensor parallel size {world_size}.")
    rank_print(f"Tensor-parallelized {tp_module_count} Linear modules.")
    if missing:
        rank_print(f"Missing checkpoint tensors kept from initialization: {len(missing)}")
    if unexpected:
        rank_print(f"Unexpected checkpoint tensors ignored: {len(unexpected)}")

    return InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
        device=device,
    )


def main():
    args = parse_args()
    rank, local_rank, world_size, device = init_tensor_parallel(args.num_gpus, args.allow_non_h100)
    set_seed(args.seed)

    inferencer = load_inferencer(args.model_path, device, rank, world_size, args.tp_plan)
    image = pil_img2rgb(Image.open(args.input_image))

    result = inferencer(
        image=image,
        text=args.prompt,
        cfg_text_scale=args.cfg_text_scale,
        cfg_img_scale=args.cfg_img_scale,
        cfg_interval=[0.0, 1.0],
        timestep_shift=args.timestep_shift,
        num_timesteps=args.num_timesteps,
        cfg_renorm_min=args.cfg_renorm_min,
        cfg_renorm_type=args.cfg_renorm_type,
    )

    if rank == 0:
        os.makedirs(os.path.dirname(args.output_image) or ".", exist_ok=True)
        result["image"].save(args.output_image)
        print(f"Saved edited image to {args.output_image}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
