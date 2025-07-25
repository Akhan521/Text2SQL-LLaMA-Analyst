from src.data_loader import load_data
from src.model_loader import load_model, load_tokenizer
from src.generate import generate
import torch

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine whether to use a fine-tuned model or the base model.
    use_base = False             # Set this to 'True' if you want to use the base model.
    use_local_finetuned = False  # Set this to 'True' if you want to load your locally fine-tuned model from the 'logs' directory.

    model_path = "NousResearch/Llama-2-7b-hf" if use_base else "logs" if use_local_finetuned else "akhan365/llama2-finetuned-for-text2sql"
    dataset_name = "ChrisHayduk/Llama-2-SQL-Dataset"

    # Load the model and tokenizer.
    print(f"\nLoading model and tokenizer from {model_path}...")
    model = load_model(model_path, quantized=False) 
    tokenizer = load_tokenizer(model_path)

    # To evaluate our model's alignment with the dataset, we'll choose a sample question from the dataset.
    _, eval_dataset = load_data(dataset_name)
    sample_question = eval_dataset[0]['input']  
    correct_answer = eval_dataset[0]['output']  # Get the corresponding output to the sample question.
    
    # If you'd like to test your own prompt, you can uncomment the following line:
    # sample_question = "What is the minimum age of the employees in the company?"

    print(f'\nGenerating text using {'fine-tuned' if not use_base else 'base'} model...')
    print('=' * 50)
    print(f'Prompt: "{sample_question}"\n')
    generated_text = generate(sample_question, model, tokenizer, device)
    print(f'\nGenerated text: "{generated_text}"\n')
    print(f'\nExpected answer: "{correct_answer}"\n')

