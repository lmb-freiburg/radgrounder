import os
import torch



def load_siglip_weights_into_paligemma(paligemma, siglip_model_path=None):
    # An empty/None config value falls back to SIGLIP_CKPT_PATH (which defaults to the
    # staged models/siglip/siglip_refrad2d_v18.ckpt) — keeps machine paths out of the config.
    if not siglip_model_path:
        from radgrounder.paths import SIGLIP_CKPT_PATH
        siglip_model_path = SIGLIP_CKPT_PATH
    if not siglip_model_path:
        raise ValueError(
            "No SigLIP checkpoint provided. Set the `siglip_model_path` config key "
            "or the SIGLIP_CKPT_PATH environment variable."
        )
    if not os.path.exists(siglip_model_path):
        raise FileNotFoundError(
            f"SigLIP checkpoint not found at: {siglip_model_path}\n"
            "Set SIGLIP_CKPT_PATH (or the config's siglip_model_path) to a valid checkpoint, "
            "or set load_siglip_weights=false to train with PaliGemma-2's stock vision tower."
        )
    print(f"Loading SigLIP weights from: {siglip_model_path}")

    #laod the model using torch
    siglip_model = torch.load(siglip_model_path, map_location="cpu")
    vision_tower_params = {key: p for key, p in siglip_model["state_dict"].items() if "visual" in key}

    # print("First 10 parameters in SigLIP model:")
    # for i , (name, param) in enumerate(vision_tower_params.items()):
    #     if "visual" in name:
    #         print(name, param.numel())
    #         if i == 10:
    #             break
            
    total_siglip_vision_tower_params = sum(p.numel() for k, p in vision_tower_params.items())
    print(f"Total parameters in new SigLIP vision tower: {total_siglip_vision_tower_params}")

    for name, param in list(vision_tower_params.items()):
        new_name = name.replace("model.visual.trunk.blocks", "vision_model.encoder.layers")
        if "trunk.attn_pool" in name:
            vision_tower_params.pop(name)
            continue
        
        if "model.visual.trunk.pos_embed" in name:
            vision_tower_params.pop(name)
            new_name = new_name.replace("model.visual.trunk.pos_embed", "vision_model.embeddings.position_embedding.weight")
            param = param[0]
            vision_tower_params[new_name] = param
        elif "model.visual.trunk.norm" in name:
            vision_tower_params.pop(name)
            new_name = new_name.replace("model.visual.trunk.norm", "vision_model.post_layernorm")
            vision_tower_params[new_name] = param
        # elif "model.visual.trunk.attn_pool.norm" in name:
        #     vision_tower_params.pop(name)
        #     new_name = new_name.replace("model.visual.trunk.attn_pool.norm", "vision_model.post_layernorm")
            vision_tower_params[new_name] = param
        elif "qkv" in name:
            # Split qkv into q, k, v
            # weight_type = "weight" if "weight" in name else "bias"
            qkv = param
            dim = qkv.shape[0] // 3
            vision_tower_params[new_name.replace("attn.qkv", f"self_attn.q_proj")] = qkv[:dim]
            vision_tower_params[new_name.replace("attn.qkv", f"self_attn.k_proj")] = qkv[dim:2*dim]
            vision_tower_params[new_name.replace("attn.qkv", f"self_attn.v_proj")] = qkv[2*dim:]
            vision_tower_params.pop(name)
        elif "norm" in name:
            vision_tower_params.pop(name)
            new_name = new_name.replace("norm", "layer_norm")
            vision_tower_params[new_name] = param
        elif "proj" in name and "attn" in name:
            vision_tower_params.pop(name)
            new_name = new_name.replace("attn.proj", "self_attn.out_proj")
            vision_tower_params[new_name] = param
        else:
            new_name = new_name.replace("model.visual.trunk.patch_embed.proj", "vision_model.embeddings.patch_embedding")
        
            vision_tower_params.pop(name)
            vision_tower_params[new_name] = param
        

    paligemma.model.vision_tower.load_state_dict(vision_tower_params, strict=True)
    return paligemma


if __name__ == "__main__":
    #load normal paligemma and change the visual encoder weights to siglip weights
    from transformers import PaliGemmaForConditionalGeneration
    MODEL_ID = "google/paligemma2-3b-pt-224"
    cache_dir = None
    torch_dtype = torch.float16
    # paligemma = PaliGemmaForConditionalGeneration.from_pretrained(
    #     MODEL_ID, 
    #     torch_dtype=torch_dtype, 
    #     device_map="auto", 
    #     attn_implementation='eager', 
    #     cache_dir=cache_dir
    # )
    
    #conda activate monai_ggemma
    from modeling_groundedgemma import GroundedGemmaForConditionalGeneration
    
    paligemma = GroundedGemmaForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch_dtype,
            device_map="auto",
            attn_implementation='eager',
            cache_dir=cache_dir,
            offload_folder=None,
            offload_state_dict=False
        )

    # print("PaliGemma vision_config:")
    # print(paligemma.config.vision_config)

    print("First 10 parameters in PaliGemma vision tower:")
    #check parameter count of paligemma vision tower
    for i, (name, param) in enumerate(paligemma.model.vision_tower.named_parameters()):
        print(name, param.numel())
        if i == 10:
            break
    total_params = sum(p.numel() for p in paligemma.model.vision_tower.parameters())
    print(f"Total parameters in PaliGemma vision tower: {total_params}")
    # # --- Infer SigLIP checkpoint shapes ---
    # siglip_state = siglip_model["state_dict"]
    # patch_w = siglip_state.get("model.visual.trunk.patch_embed.proj.weight")
    # pos_embed = siglip_state.get("model.visual.trunk.pos_embed")
    # if patch_w is not None:
    #     out_dim, in_dim, k_h, k_w = patch_w.shape
    #     print(f"SigLIP patch embedding weight shape: {patch_w.shape}")
    #     print(f"SigLIP inferred patch size: {k_h}x{k_w}, embed dim: {out_dim}, in_channels: {in_dim}")
    # if pos_embed is not None:
    #     print(f"SigLIP pos_embed shape: {pos_embed.shape}")
    # paligemma.model.vision_tower.load_state_dict({k.replace("model.vision_tower.", ""): v for k, v in siglip_model["state_dict"].items() if "model.vision_tower." in k})

    # from transformers import AutoModel
    # model = AutoModel.from_pretrained("google/siglip-so400m-patch14-224")
    # #printthe model paramter count and the first 10 parameters
    # vision_tower_params = {name: p for name, p in model.named_parameters() if "vision" in name}
    # # print("First 10 parameters in SigLIP model from transformers:")
    # for i, (name, param) in enumerate(vision_tower_params.items()):
    #     print(name, param.numel())
    #     if i == 10:
    #         break
    # total_params_transformers = sum(p.numel() for p in vision_tower_params.values())
    # print(f"Total parameters in SigLIP model from transformers: {total_params_transformers}")
    load_siglip_weights_into_paligemma(paligemma)

# vision_model.encoder.layers.0.self_attn.v_proj.weight
# vision_model.encoder.layers.0.self_attn.v_proj.weight