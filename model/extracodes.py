def create_model(
        model_name: str,
        img_size: int,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = True,   # force BiomedCLIP to always use CustomTextCLIP
        force_patch_dropout: Optional[float] = None,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        output_dict: Optional[bool] = None,
        require_pretrained: bool = False,
        adapter = False,
):

    model_name = model_name.replace('/', '-')  
    checkpoint_path = None
    model_cfg = None

    if isinstance(device, str):
        device = torch.device(device)

    model_cfg = model_cfg or get_model_config(model_name)
    if model_cfg is not None:
        print(f'Loaded {model_name} model config for BiomedCLIP.')
    else:
        raise RuntimeError(f'Model config for {model_name} not found.')

    if force_quick_gelu:
        model_cfg["quick_gelu"] = True

    if force_patch_dropout is not None:
        model_cfg["vision_cfg"]["patch_dropout"] = force_patch_dropout

    if force_image_size is not None:
        model_cfg["vision_cfg"]["image_size"] = force_image_size

    cast_dtype = get_cast_dtype(precision)
    
            # ✅ FIXED indentation starts here
    model_pre = load_biomedclip_model(
        name=_MODEL_CKPT_PATHS[model_name],
        precision=precision,
        device=device,
        cache_dir=None,
        jit=jit
    )
    state_dict = model_pre.state_dict()

        # to always output dict even if it is clip
    if output_dict and hasattr(model_pre, "output_dict"):
        model_pre.output_dict = True


    


    # Always use CustomTextCLIP for BiomedCLIP
    model = CustomTextCLIP(**model_cfg, cast_dtype=cast_dtype)

    pretrained_loaded = False
    if pretrained:
        checkpoint_path = _MODEL_CKPT_PATHS.get(model_name, None)
        if checkpoint_path:
            print(f'Loading pretrained {model_name} weights ({pretrained}).')
            load_checkpoint(model, checkpoint_path)
            pretrained_loaded = True
        else:
            raise RuntimeError(f'Pretrained weights ({pretrained}) not found for model {model_name}.')

    if require_pretrained and not pretrained_loaded:
        raise RuntimeError(
            f'Pretrained weights were required for (model: {model_name}, pretrained: {pretrained}) but not loaded.'
        )

    model.to(device=device)
    if precision in ("fp16", "bf16"):
        convert_weights_to_lp(model, dtype=torch.bfloat16 if precision == 'bf16' else torch.float16)

    model.visual.image_mean = (0.48145466, 0.4578275, 0.40821073)
    model.visual.image_std = (0.26862954, 0.26130258, 0.27577711)

    if output_dict and hasattr(model, "output_dict"):
        model.output_dict = True

    if jit:
        model = torch.jit.script(model)

    return model
