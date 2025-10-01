from dataclasses import dataclass
from typing import Union, Tuple, Optional
import torch
import torch.nn as nn  # <-- add this line
import timm
import torch.nn as nn
from typing import Optional




#configurations of vision and text classes 

@dataclass
class CLIPVisionCfg:
    # Base transformer settings (ViT-B/16)
    layers: Union[Tuple[int, int, int, int], int] = 12   # ViT-B has 12 encoder layers
    width: int = 768                                     # Hidden size for ViT-B
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224

    # Additional ViT options (kept consistent with OpenCLIP, but not all used in BiomedCLIP)
    ls_init_value: Optional[float] = None
    patch_dropout: float = 0.0   # BiomedCLIP doesn’t use patch dropout (training was frozen/pretrained)
    input_patchnorm: bool = False
    global_average_pool: bool = False
    attentional_pool: bool = False
    n_queries: int = 256
    attn_pooler_heads: int = 8

    # timm-specific configs (BiomedCLIP relies on timm ViT-B/16 backbone)
    timm_model_name: str = "vit_base_patch16_224"
    timm_model_pretrained: bool = False
    timm_pool: str = ""          # no pooling used
    timm_proj: str = "linear"    # linear projection
    timm_proj_bias: bool = False
    timm_drop: float = 0.0
    timm_drop_path: Optional[float] = None

    # Output config
    output_tokens: bool = True

@dataclass
class CLIPTextCfg:
    # custom BiomedCLIP config
    context_length: int = 256
    vocab_size: Optional[int] = None   # will be inferred from HuggingFace tokenizer
    width: int = 512
    heads: Optional[int] = None        # handled by HF model (BiomedBERT)
    layers: Optional[int] = None       # also from HF model, not manually set
    ls_init_value: Optional[float] = None  
    
    # HuggingFace configs
    hf_model_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    hf_tokenizer_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    hf_model_pretrained: bool = True   # we load pretrained BiomedBERT weights
    
    # projection and pooling
    proj: str = "mlp"                  # as given in config
    pooler_type: str = "cls_last_hidden_state_pooler"
    
    # token embedding options
    embed_cls: bool = False
    pad_id: int = 0
    output_tokens: bool = False

def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype


def build_model_from_biomedclip_state_dict(
    state_dict: dict,
    quick_gelu=True,
    cast_dtype=torch.float16,
):
    # Detect BiomedCLIP vision backbone
    vit = any(k.startswith("visual.trunk.patch_embed") for k in state_dict.keys())

    if vit:
        # Extract ViT parameters
        patch_key = [k for k in state_dict if "visual.trunk.patch_embed.proj.weight" in k][0]
        vision_patch_size = state_dict[patch_key].shape[-1]
        vision_width = state_dict["visual.trunk.blocks.0.attn.qkv.weight"].shape[1]

        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.trunk.blocks.") and k.endswith(".attn.qkv.weight")])

        pos_key = [k for k in state_dict if "visual.trunk.pos_embed" in k]
        if pos_key:
            grid_size = round((state_dict[pos_key[0]].shape[1] - 1) ** 0.5)
            image_size = vision_patch_size * grid_size
        else:
            # fallback if no explicit pos_embed
            image_size = 224

        print("=== Vision Transformer (ViT) Parameters ===")
        print(f"Patch size       : {vision_patch_size}")
        print(f"Hidden width     : {vision_width}")
        print(f"Encoder layers   : {vision_layers}")
        print(f"Image size       : {image_size}x{image_size}")
    else:
        raise ValueError("Could not detect ViT backbone in BiomedCLIP state_dict")

    # === Text encoder (BERT-based) ===
    vocab_size = state_dict["text.transformer.embeddings.word_embeddings.weight"].shape[0]
    context_length = state_dict["text.transformer.embeddings.position_embeddings.weight"].shape[0]
    transformer_width = state_dict["text.transformer.embeddings.word_embeddings.weight"].shape[1]
    transformer_layers = len([
        k for k in state_dict.keys()
        if k.startswith("text.transformer.encoder.layer.") and k.endswith(".output.LayerNorm.weight")
    ])
    transformer_heads = state_dict["text.transformer.encoder.layer.0.attention.self.query.weight"].shape[0] // transformer_width

    print("\n=== Text Transformer (BERT) Parameters ===")
    print(f"Vocab size       : {vocab_size}")
    print(f"Context length   : {context_length}")
    print(f"Hidden width     : {transformer_width}")
    print(f"Encoder layers   : {transformer_layers}")
    print(f"Attention heads  : {transformer_heads}")

    # Vision config
    vision_cfg = CLIPVisionCfg(
        layers=vision_layers,
        width=vision_width,
        patch_size=vision_patch_size,
        image_size=image_size,
    )
    print("\n=== BiomedCLIP Vision Config ===\n")
    print(f"Layers: {vision_cfg.layers}")
    print(f"Width: {vision_cfg.width}")
    print(f"Patch size: {vision_cfg.patch_size}")
    print(f"Image size: {vision_cfg.image_size}")

    # Text config
    text_cfg = CLIPTextCfg(
        context_length=context_length,
        vocab_size=vocab_size,
        width=transformer_width,
        heads=transformer_heads,
        layers=transformer_layers,
    )

    print("\n === BiomedCLIP Text Config === \n")
    print(f"Context length   : {text_cfg.context_length}")
    print(f"Vocab size       : {text_cfg.vocab_size}")
    print(f"Hidden width     : {text_cfg.width}")
    print(f"Attention heads  : {text_cfg.heads}")
    print(f"Encoder layers   : {text_cfg.layers}")


    embed_dim = state_dict["visual.head.proj"].shape[1] if "visual.head.proj" in state_dict else 512
    print(f"\nProjection dimension (embed_dim): {embed_dim}")