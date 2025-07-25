from src.data_loader import load_data, preprocess_dataset
from src.model_loader import load_model, load_tokenizer
from src.generate import generate
import torch

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine whether to use a fine-tuned model or the base model.
    use_base = False             # Set this to 'True' if you want to use the base model.
    use_local_finetuned = False  # Set this to 'True' if you want to load your locally fine-tuned model from the 'logs' directory.

    model_path = "NousResearch/Llama-2-7b-hf" if use_base else "logs" if use_local_finetuned else "akhan365/llama2-finetuned-for-text2sql"

    # Load the model and tokenizer.
    print(f"\nLoading model and tokenizer from {model_path}...")
    model = load_model(model_path, quantized=False) 
    tokenizer = load_tokenizer(model_path)

    # Play around with the prompt you want to generate text for.
    prompt = "The results are "

    print(f'\nGenerating text using {'fine-tuned' if not use_base else 'base'} model...')
    print('=' * 50)
    print(f'Prompt: "{prompt}"\n')
    generated_text = generate(prompt, model, tokenizer, device)
    print(f'\nGenerated text: "{generated_text}"\n')

