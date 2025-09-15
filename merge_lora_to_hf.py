import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_lora_to_hf(base_model: str, adapter_dir: str, output_dir: str):

    print(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(base_model)
    model = PeftModel.from_pretrained(model, adapter_dir)

    print("Merging LoRA adapters into base model...")
    model = model.merge_and_unload()


    print(f"Saving merged model to: {output_dir}")
    model.save_pretrained(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    print("tokenizer.chat_template",tokenizer.chat_template)
    tokenizer.save_pretrained(output_dir)

    print("done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model into HF-compatible format.")
    parser.add_argument("--base_model", type=str, required=True, help="Base model name or path (e.g. EleutherAI/pythia-31m)")
    parser.add_argument("--adapter_dir", type=str, required=True, help="Dir with trained LoRA adapters (contains adapter_config.json)")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save the merged Hugging Face model")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    merge_lora_to_hf(args.base_model, args.adapter_dir, args.output_dir)
