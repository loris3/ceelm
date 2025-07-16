import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_path):
    print(f"Loading model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",  # ✅ Directly map to available GPU(s)
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    return tokenizer, model

def chat_loop(tokenizer, model):
    print("\nModel is ready. Type your prompt and press Enter.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        prompt = input("You: ")
        if prompt.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                # max_length=200,
                do_sample=False,
                # temperature=0.7,
                # top_p=0.9,
                num_return_sequences=1,
            )

        response = tokenizer.decode(output[0], skip_special_tokens=True)
        print("Model:", response[len(prompt):].strip(), "\n")

if __name__ == "__main__":
    model_dir = "/srv/home/users/loriss21cs/cfe/out/OLMo-2-1124-7B-math-lora-gas8-bs4-mezo-v_proj-2560_largest_grad-5steps-seed0"
    tokenizer, model = load_model(model_dir)
    chat_loop(tokenizer, model)
