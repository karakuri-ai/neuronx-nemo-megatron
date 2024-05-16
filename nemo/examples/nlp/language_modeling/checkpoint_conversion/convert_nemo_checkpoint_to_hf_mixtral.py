import os
import argparse
import json
from pathlib import Path
from os.path import join
from glob import glob
import re
import torch
import torch_xla.utils.serialization as xser
from transformers import AutoConfig, AutoModelForCausalLM


def fix_query_key_value_ordering(param, checkpoint_version, num_splits, num_heads, hidden_size):
    # Permutes layout of param tensor to [num_splits * num_heads * hidden_size, :]
    # for compatibility with later versions of NVIDIA Megatron-LM.
    # The inverse operation is performed inside Megatron-LM to read checkpoints:
    # https://github.com/NVIDIA/Megatron-LM/blob/v2.4/megatron/checkpointing.py#L209
    # If param is the weight tensor of the self-attention block, the returned tensor
    # will have to be transposed one more time to be read by HuggingFace GPT2.
    input_shape = param.size()
    if checkpoint_version == 1.0:
        # version 1.0 stores [num_heads * hidden_size * num_splits, :]
        saved_shape = (num_heads, hidden_size, num_splits) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 2)
        param = param.transpose(1, 2).contiguous()
    elif checkpoint_version >= 2.0:
        # other versions store [num_heads * num_splits * hidden_size, :]
        saved_shape = (num_heads, num_splits, hidden_size) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 1).contiguous()
    param = param.view(*input_shape)
    return param

def get_tp_pp_degree(path_to_checkpoints):
    TP = 1
    PP = 1
    for folder in os.listdir(path_to_checkpoints):
        pp_search = re.search('pp_rank_[\d]*', folder)
        if pp_search:
            PP = max(PP, 1+int(pp_search[0].split('pp_rank_')[1]))
        if PP>1:
            tp_search = re.search('tp_rank_[\d]*', folder)
            if tp_search:
                TP = max(TP, 1+int(tp_search[0].split('tp_rank_')[1]))
        else:
            tp_search = re.search('mp_rank_[\d]*', folder)
            if tp_search:
                TP = max(TP, 1+int(tp_search[0].split('mp_rank_')[1]))

    return TP, PP

def _get_tp_str(tp: int):
    tp_template = '00'
    tp = str(tp)
    leading_zeros = len(tp_template) - len(tp)
    return ''.join(['0'] * leading_zeros + list(tp))

def _get_pp_str(pp: int):
    pp_template = '000'
    pp = str(pp)
    leading_zeros = len(pp_template) - len(pp)
    return ''.join(['0'] * leading_zeros + list(pp))


def load_model(path, is_xser=False, dtype=None):
    if is_xser:
        model = xser.load(path)['state_dict']
    else:
        model = torch.load(path)['state_dict']

    if dtype:
        torch_dtype = getattr(torch, dtype)
        model = {k: v.to(torch_dtype) for k, v in model.items()}

    return model


def get_checkpoints_for_pp(pp: int, path_to_checkpoints: str, PP: int=1, TP: int=1, is_xser: bool=False, step=None, dtype=None):
    """
    Returns all checkpoints for specified PP rank
    """
    if PP == 1 and TP == 1:
        pp_str = ""
    else:
        pp_str = f'tp_rank_*_pp_rank_{_get_pp_str(pp)}' if PP > 1 else "mp_rank_*"

    template = join(path_to_checkpoints, pp_str, '*.ckpt')

    try:
        # take largest step saved model from the available checkpoints
        if step is None:
            step = max(
                [int(re.match(r".*megatron_mixtral--step=(\d+).*ckpt$", i).group(1))
                for i in glob(template)])
        template = join(path_to_checkpoints, pp_str, f'*megatron_mixtral--step={step}-*.ckpt')
    except AttributeError:
        ...

    tp_paths = sorted(glob(template))
    return {i: load_model(p, is_xser=is_xser, dtype=dtype) for i, p in enumerate(tp_paths)}


def _get_nemo_key(k, nemo_key = 'model.language_model.'):
    if "final_layernorm" in k:
        nemo_key += 'encoder.'
    return k.replace(nemo_key, '')

def convert_checkpoint(config_file,
                       path_to_checkpoints,
                       output_path,
                       checkpoint_version=2.0,
                       is_xser=False,
                       step=None,
                       dtype=None):

    with open(config_file, "r") as f:
        config = json.load(f)

    translation = {
        "embedding.word_embeddings.weight": (1, "model.embed_tokens.weight", 0, 0),
        "input_layernorm.weight": (0, "input_layernorm.weight", None, 0),
        "self_attention.query.weight": (1, "self_attn.query.weight", 0, 0),
        "self_attention.key_value.weight": (1, "self_attn.key_value.weight", 0, 0),
        "self_attention.dense.weight": (1, "self_attn.o_proj.weight", 1, 0),
        "post_attention_layernorm.weight": (0, "post_attention_layernorm.weight", None, 0),
        "self_attention.core_attention.rotary_emb.inv_freq": (0, "self_attn.rotary_emb.inv_freq", None, 0),
        "mlp.router.gate.weight": (1, "block_sparse_moe.gate.weight", 1, 0),
        "final_layernorm.weight": (0, "model.norm.weight", None, 0),
        "output_layer.weight": (1, "lm_head.weight", 0, 0),  # this is shared
    }
    for i in range(config["num_local_experts"]):
        # gate_proj
        translation[f"mlp.experts.{i}.dense_h_to_4h.weight"] = (1, f"block_sparse_moe.experts.{i}.w1.weight", 0, 0)
        # down_proj
        translation[f"mlp.experts.{i}.dense_h_to_4h_2.weight"] = (1, f"block_sparse_moe.experts.{i}.w3.weight", 0, 0)
        # up_proj
        translation[f"mlp.experts.{i}.dense_4h_to_h.weight"] = (1, f"block_sparse_moe.experts.{i}.w2.weight", 1, 0)

    nemo_key = "model.language_model."
    br_key = "model.layers."

    TP, PP = get_tp_pp_degree(path_to_checkpoints)
    print(f"TP: {TP}, PP: {PP}")

    attn_heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    hidden_size_per_head = config.get("head_dim") or (config["hidden_size"] // attn_heads)

    hf_model = {}

    layer_re = re.compile("model.language_model.encoder.layers\.(\d+)\.([a-z0-9_.]+)\.([a-z_]+)")

    for pp in range(PP):
        print(f"Loading PP={pp}")
        tp_models = get_checkpoints_for_pp(pp, path_to_checkpoints, PP, TP, is_xser, step, dtype)
        layer_keys = tp_models[0].keys()
        for k in layer_keys:
            print(f">> {k}")
            if "position_embeddings" in k:
                nemo_key = _get_nemo_key(k)
                _, key, _, _ = translation[nemo_key]
                hf_model[key] = tp_models[0][k]
                continue

            if "word_embeddings" in k:
                nemo_key = _get_nemo_key(k)
                split, key, dim, transpose = translation[nemo_key]
                hf_model[key] = torch.concat([tp_models[i][k] for i in range(len(tp_models))], dim=0)
                continue

            if "output_layer" in k:
                nemo_key = _get_nemo_key(k)
                split, key, dim, transpose = translation[nemo_key]
                hf_model[key] = torch.concat([tp_models[i][k] for i in range(len(tp_models))], dim=dim)
                continue

            if "final_layernorm" in k:
                nemo_key = _get_nemo_key(k)
                split, key, dim, transpose = translation[nemo_key]
                hf_model[key] = tp_models[0][k]
                continue

            m = layer_re.match(k)
            layer_idx = m.group(1)
            op_name = m.group(2)
            weight_or_bias = m.group(3)
            nemo_key = f"{op_name}.{weight_or_bias}"
            split, key, dim, transpose = translation[nemo_key]
            ln_idx= int(layer_idx) + pp*(config["num_hidden_layers"]//PP)
            hf_key = f"{br_key}{ln_idx}.{key}"
            if split:
                hf_model[hf_key] = torch.concat([tp_models[i][k] for i in range(len(tp_models))], dim=dim)
            else:
                hf_model[hf_key] = tp_models[0][k]
            if "query" in k:
                hf_model[hf_key] = fix_query_key_value_ordering(hf_model[hf_key], checkpoint_version, 1, attn_heads, hidden_size_per_head)
            if "key_value" in k:
                hf_model[hf_key] = fix_query_key_value_ordering(hf_model[hf_key], checkpoint_version, 2, kv_heads, hidden_size_per_head)
            if transpose:
                hf_model[hf_key] = torch.transpose(hf_model[hf_key], 0, 1)

            # Break Q K V into three matrices
            if "query" in k:
                hf_key_q = f"{br_key}{ln_idx}.self_attn.q_proj.weight"
                hf_model[hf_key_q] = hf_model[hf_key]
                hf_model.pop(hf_key)
            if "key_value" in k:
                hf_key_k = f"{br_key}{ln_idx}.self_attn.k_proj.weight"
                hf_key_v = f"{br_key}{ln_idx}.self_attn.v_proj.weight"
                size_per_seg = hf_model[hf_key].shape[0] // 2
                hf_model[hf_key_k], hf_model[hf_key_v] = torch.split(hf_model[hf_key], size_per_seg, dim=0)
                hf_model.pop(hf_key)

    path = Path(output_path)
    path.mkdir(parents=True, exist_ok=True)
    config = AutoConfig.from_pretrained(config_file)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    model.load_state_dict(hf_model, assign=True)
    model.save_pretrained(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_version", default=2.0)
    parser.add_argument(
        "--path_to_checkpoints",
        type=str,
        help="Path to the checkpoints from creating NeMo checkpoint files using `convert_hf_checkpoint_to_nemo.py`",
        required=True
    )
    parser.add_argument(
        "--config_file",
        type=str,
        help="The config json file describing the pre-trained model.",
        required=True
    )
    parser.add_argument(
        "--output_path",
        default="",
        type=str,
        help="output path",
    )
    parser.add_argument(
        "--is_xser",
        default=False,
        type=bool
    )
    parser.add_argument(
        "--step",
        default=None,
        type=int
    )
    parser.add_argument(
        "--dtype",
        default=None,
        type=str
    )
    args = parser.parse_args()
    convert_checkpoint(args.config_file, args.path_to_checkpoints, args.output_path, args.checkpoint_version, args.is_xser, args.step, args.dtype)
