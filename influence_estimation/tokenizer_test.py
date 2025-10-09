import os
from datasets import load_dataset
from transformers import AutoTokenizer
from util import tokenize_dataset  

def main():
  
    tokenizer_name = "allenai/OLMo-2-0425-1B-SFT"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    dataset_name = "loris3/tulu3-test-small"
    dataset = load_dataset(dataset_name, split="train")

    print(f"Loaded dataset with {len(dataset)} examples")

    tokenized_dataset = tokenize_dataset(
        dataset,
        tokenizer=tokenizer,
        max_length=512,
        chat_template_path="./chat_template.jinja",  
        assistant_only_loss=True,
        text_column="messages",  
        num_proc=20,
        re_index=True
    )

    # Print a single example to verify
    print("Tokenized example:")
    print(tokenized_dataset[111:113])

if __name__ == "__main__":
    main()
