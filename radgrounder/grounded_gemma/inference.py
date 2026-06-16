import os
import torch
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True
# torch._dynamo.disable()
# os.environ["FLASH_ATTENTION_2_DISABLED"] = "1"
from transformers import PaliGemmaProcessor, AutoConfig
import torch
from modeling_groundedgemma import GroundedGemmaForConditionalGeneration
from transformers.image_utils import load_image
import os

torch._dynamo.config.disable = True
torch._dynamo.config.suppress_errors = True


if __name__ == "__main__":
    # MODEL_ID = "google/paligemma2-3b-pt-224"
    # MODEL_ID = "google/paligemma2-3b-mix-224"
    cache_dir = "../paligemma/models/paligemma-3b"
    MODEL_ID = os.environ.get("MODEL_PATH", "")

    processor = PaliGemmaProcessor.from_pretrained(MODEL_ID, cache_dir=cache_dir, use_fast=True)

    TORCH_DTYPE = torch.bfloat16

    print(f"Loading model from {MODEL_ID}")
    config = AutoConfig.from_pretrained(MODEL_ID,
                                        cache_dir=cache_dir,
                                        device_map="auto",
                                        attn_implementation='eager',
                                        torch_dtype=TORCH_DTYPE,
                                       )
    
    #adjust the num of image tokens for the unimedclip model
    # model = GroundedGemmaForConditionalGeneration(config)
    model = GroundedGemmaForConditionalGeneration.from_pretrained(MODEL_ID, cache_dir=cache_dir, torch_dtype=TORCH_DTYPE, device_map="auto").eval()
    # model = model.to(dtype=TORCH_DTYPE, device='cuda').eval()

    #count number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {num_params}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")

    exit()
    # from transformers import PaliGemmaForConditionalGeneration
    # model = PaliGemmaForConditionalGeneration.from_pretrained(MODEL_ID, cache_dir=cache_dir, torch_dtype=TORCH_DTYPE, device_map="auto").eval()
    # model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    # model.eval()



    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
    image = load_image(url)
    # dummy_input_text = "<image> What do you see in this image? <p>kidney</p>."
    dummy_input_text = "<image> Which structures are visible in the image?"
    prompt = [dummy_input_text, dummy_input_text]
    images = [image, image]

    special_tokens = ["<p>", "</p>", "id="]
    processor.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    end_special_token_id = processor.tokenizer.convert_tokens_to_ids("</p>")
    model.set_seg_token_id(end_special_token_id)
    print(f"Special token ID, </p>: {end_special_token_id}")

    model_inputs = processor(text=prompt, images=images, return_tensors="pt")
    for k, v in model_inputs.items():
        if k == "pixel_values":
            model_inputs[k] = v.to(model.device, dtype=torch.bfloat16)
        else:
            model_inputs[k] = v.to(model.device)
    # print(f"Input keys: {model_inputs.keys()}")
    # print(f"Input shapes: {[v.shape for v in model_inputs.values()]}")
    model_inputs["segment_gt"] = torch.rand(len(prompt), 224, 224, dtype=torch.float64).to(model.device)  # Dummy segmentation mask
    # model_inputs["seg_token_pos"] = torch.tensor([[0, 8], [1, 8]], dtype=torch.long).to(model.device)  # Dummy segmentation token positions
    input_len = model_inputs["input_ids"].shape[-1]
    print("Input IDs shape:", model_inputs["input_ids"].shape)
    print(f"Prompt {prompt}")
    with torch.inference_mode():
        generation, segmentation_logits_and_pos = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
        generation = generation[0][input_len:]
        decoded = processor.decode(generation, skip_special_tokens=True)
        print(decoded)
        print(f"Segment logits shape: {len(segmentation_logits_and_pos)}")

