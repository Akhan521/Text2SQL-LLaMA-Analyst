from src.data_loader import load_data, preprocess_dataset
from src.model_loader import load_model, load_tokenizer
from src.generate import generate
import torch

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "NousResearch/Llama-2-7b-hf"

    print(f"Loading model and tokenizer for {model_name}...")
    model = load_model(model_name, quantized=False) 
    tokenizer = load_tokenizer(model_name)

    prompt = "The results are "

    print(f"\nGenerating text for prompt: {prompt}")
    print("=" * 50)
    generated_text = generate(prompt, model, tokenizer, device)
    print(f"\nGenerated text: {generated_text}\n")

