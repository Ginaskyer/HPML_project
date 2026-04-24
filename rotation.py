import torch
import torch.nn as nn

from tqdm import tqdm

def load_or_create_R1(mode: str,
                      device: str,
                      save_dir: str = None,
                      dim: int = None):
    if mode == "offline":
        pre_compute_r1 = torch.load(save_dir, weights_only=False, map_location="cpu")
        pre_compute_dim = pre_compute_r1["dim"]
        R1 = nn.utils.parametrizations.orthogonal(nn.Linear(pre_compute_dim, pre_compute_dim, bias=False, dtype=torch.float32),
                                                  orthogonal_map="cayley")
        R1.load_state_dict(pre_compute_r1["r1"])

        return R1.to(device)
    elif mode == "online":
        assert dim is not None, "dim must be specified"

        R1 = nn.utils.parametrizations.orthogonal(nn.Linear(dim, dim, bias=False, dtype=torch.float32),
                                                  orthogonal_map="cayley")

        return R1.to(device)
    else:
        raise ValueError("Unknown mode")

def rotate_attention_output(layer, rotation_cache) -> None:
    # Rotate output matrix of the self-attention layer.
    W = layer.self_attn.o_proj
    dtype = W.weight.data.dtype
    device = W.weight.data.device

    R1 = rotation_cache[str(device)]
    W_ = W.weight.data.to(device=device, dtype=torch.float64)
    R1 = R1.to(device=device, dtype=torch.float64)
    W.weight.data = torch.matmul(R1.T, W_).to(device=device, dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=device, dtype=torch.float64)
        W.bias.data = torch.matmul(R1.T, b).to(device=device, dtype=dtype)
        del b
    del W_


def rotate_attention_inputs(layer, rotation_cache) -> None:
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        dtype = W.weight.dtype
        device = W.weight.device

        R1 = rotation_cache[str(device)]

        W_ = W.weight.to(device=device, dtype=torch.float64)
        R1 = R1.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, R1).to(device=device, dtype=dtype)
    del W_


def rotate_mlp_input(layer, rotation_cache):
    mlp_inputs = []
    mlp_inputs.append(layer.mlp.up_proj)
    mlp_inputs.append(layer.mlp.gate_proj)

    for W in mlp_inputs:
        dtype = W.weight.dtype
        device = W.weight.device

        R1 = rotation_cache[str(device)]

        W_ = W.weight.data.to(device=device, dtype=torch.float64)
        R1 = R1.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, R1).to(device=device, dtype=dtype)
        del W_


def rotate_mlp_output(layer, rotation_cache):
    # Rotate the MLP output weights and bias.
    W = []
    W.append(layer.mlp.down_proj)

    if isinstance(W, list):
        for w in W:
            dtype = w.weight.data.dtype
            device = w.weight.data.device

            R1 = rotation_cache[str(device)]

            W_ = w.weight.data.to(device=device, dtype=torch.float64)
            R1 = R1.to(device=device, dtype=torch.float64)
            w.weight.data = torch.matmul(R1.T, W_).to(device=device, dtype=dtype)
            # apply_exact_had_to_linear(w, had_dim=-1, output=False) #apply exact (inverse) hadamard on the weights of mlp output
            if w.bias is not None:
                b = w.bias.data.to(device=device, dtype=torch.float64)
                w.bias.data = torch.matmul(R1.T, b).to(device=device, dtype=dtype)
                del b
            del W_

    else:
        dtype = W.weight.data.dtype
        device = W.weight.data.device

        R1 = rotation_cache[str(device)]

        W_ = W.weight.data.to(device=device, dtype=torch.float64)
        R1 = R1.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(R1.T, W_).to(device=device, dtype=dtype)
        # apply_exact_had_to_linear(W, had_dim=-1, output=False) #apply exact (inverse) hadamard on the weights of mlp output
        if W.bias is not None:
            b = W.bias.data.to(device=device, dtype=torch.float64)
            W.bias.data = torch.matmul(R1.T, b).to(device=device, dtype=dtype)
            del b
        del W_


def rotate_embeddings(model, rotation_cache) -> None:
    # Rotate the embeddings.
    W = model.model.embed_tokens
    dtype = W.weight.data.dtype
    device = W.weight.data.device

    R1 = rotation_cache[str(device)]

    W_ = W.weight.data.to(device=device, dtype=torch.float64)
    R1 = R1.to(device=device, dtype=torch.float64)
    W.weight.data = torch.matmul(W_, R1).to(device=device, dtype=dtype)
    print("embedding rotated:", W_.size())
    del W_


def rotate_head(model, rotation_cache) -> None:
    # Rotate the head.
    W = model.lm_head
    dtype = W.weight.data.dtype
    device = W.weight.data.device

    R1 = rotation_cache[str(device)]

    W_ = W.weight.data.to(device=device, dtype=torch.float64)
    R1 = R1.to(device=device, dtype=torch.float64)
    W.weight.data = torch.matmul(W_, R1).to(device=device, dtype=dtype)
    del W_


def cleanup_memory(verbos=True) -> None:
    """Run GC and clear GPU memory."""
    import gc
    import inspect
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        # if verbos:
        #     logging.info(
        #         f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
        #         f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
        #     )


def fuse_rotation(model: nn.Module, r1_rotation_cache, r2_rotation_dict):
    """
    1) For each transformer layer:
       - Fuse input RMSNorm scale into q_proj, k_proj, v_proj.
       - Fuse post-attention/post-layer RMSNorm scale into:
           * MoE: router, and every expert.{up_proj, gate_proj}
           * Non-MoE MLP: mlp.{up_proj, gate_proj}  (down_proj is after nonlinearity; do NOT fuse there)
    2) Set those RMSNorm weights to ones.
    The RMS normalization itself remains active (only the affine scale is folded).
    """

    # config = model.config
    rotate_embeddings(model, r1_rotation_cache)
    rotate_head(model, r1_rotation_cache)
    cleanup_memory()
    layers = model.model.layers

    for idx in tqdm(range(len(layers)), unit="layer", desc="Rotating"):
        layer = layers[idx]
        rotate_attention_inputs(layer, r1_rotation_cache)
        rotate_attention_output(layer, r1_rotation_cache)

        rotate_mlp_input(layer, r1_rotation_cache)
        rotate_mlp_output(layer, r1_rotation_cache)
        del layer
        cleanup_memory()


def _fuse_scale_into_linear(linear: nn.Linear, scale: torch.Tensor):
    """
    Fold an elementwise input scale (RMSNorm weight) into a Linear's weight.
    Linear.weight: (out_features, in_features)
    We need to scale columns by `scale` (shape [in_features]).
    """
    W = linear.weight.data
    if W.shape[1] != scale.numel():
        # Dimension mismatch; skip
        return
    # Broadcast scale across rows to multiply each input column
    W *= scale.unsqueeze(0).to(W.device, dtype=W.dtype)


def _set_rms_weight_ones(rms: nn.Module):
    """
    Set RMSNorm weight to ones if it exists.
    """
    w = getattr(rms, "weight", None)
    if isinstance(w, torch.Tensor):
        w.data.fill_(1.0)


def fuse_weight(model: nn.Module):
    # # Try to locate the canonical stack: model.layers
    layers = model.model.layers

    for i, layer in enumerate(layers):

        in_norm = getattr(layer, "input_layernorm")

        attn = getattr(layer, "self_attn")
        scale = in_norm.weight.detach()

        prof_lists = ["q_proj", "k_proj", "v_proj"]

        for proj_name in prof_lists:
            proj = getattr(attn, proj_name)
            _fuse_scale_into_linear(proj, scale)
        _set_rms_weight_ones(in_norm)

        # mlp
        post_norm = getattr(layer, "post_attention_layernorm")
        scale = post_norm.weight.detach()

        mlp = getattr(layer, "mlp")
        up = getattr(mlp, "up_proj")
        gate = getattr(mlp, "gate_proj")
        _fuse_scale_into_linear(up, scale)
        _fuse_scale_into_linear(gate, scale)
        _set_rms_weight_ones(post_norm)

    lang_model = model.model
    norm = getattr(lang_model, "norm")
    scale = norm.weight.detach()
    proj = getattr(model, "lm_head")
    _fuse_scale_into_linear(proj, scale)
    _set_rms_weight_ones(norm)

    torch.cuda.empty_cache()