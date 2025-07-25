from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from src.data_loader import load_data, preprocess_dataset
from src.model_loader import load_model, load_tokenizer

def train_model(model_name: str = "NousResearch/Llama-2-7b-hf", dataset_name: str = "ChrisHayduk/Llama-2-SQL-Dataset", quantized: bool = True) -> None:
    '''
    Train a specified model on a dataset from the Hugging Face Hub (e.g., Llama-2-7b-hf).

    Args:
        model_name (str): The name of the model to load. Defaults to "NousResearch/Llama-2-7b-hf".
        dataset_name (str): The name of the dataset to load. Defaults to "ChrisHayduk/Llama-2-SQL-Dataset".
        quantized (bool): Whether to load the model in quantized format. Defaults to True.

    Returns:
        None
    '''
    model = load_model(model_name, quantized)
    tokenizer = load_tokenizer(model_name)

    train_dataset, eval_dataset = load_data(dataset_name)
    train_dataset = preprocess_dataset(train_dataset, tokenizer)

    peft_config = LoraConfig(
        r = 16, # Rank for LoRA layers.
        lora_alpha = 32, # Scaling factor for LoRA layers.
        target_modules =  ['q_proj', 'k_proj', 'down_proj', 'v_proj', 'gate_proj', 'o_proj', 'up_proj'], # We target the attention and feed-forward layers.
        lora_dropout = 0.05, 
        task_type = "CAUSAL_LM"
    )

    # Prepare the model for k-bit training (for quantization).
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA to the model.
    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir = "logs",
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4, # Simulate a larger batch size.
        num_train_epochs = 1,
        learning_rate = 2e-4,
        fp16 = True,
        optim = "paged_adamw_8bit" if quantized else "adamw_torch", # Use 8-bit AdamW for quantized models.
        lr_scheduler_type = "cosine",
        warmup_ratio = 0.05, # To avoid exploding gradients.
        report_to='none'
    )
    
    trainer = Trainer(
        model = model,
        train_dataset = train_dataset,
        data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm = False), # Handles any additional data processing (no masking for causal LM).
        args = training_args,
    )

    trainer.train()  # Start training the model.
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":

    model_name = "NousResearch/Llama-2-7b-hf"
    dataset_name = "ChrisHayduk/Llama-2-SQL-Dataset"

    train_model(model_name, dataset_name, quantized=True)  # Start the training process.
    print("Training complete. Model saved to './trained_model'.")
