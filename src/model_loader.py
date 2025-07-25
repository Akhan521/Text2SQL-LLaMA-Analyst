from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import transformers
import bitsandbytes as bnb

def load_tokenizer(model_name: str = "NousResearch/Llama-2-7b-hf") -> AutoTokenizer:
    '''
    Load the tokenizer for the specified model.

    Args:
        model_name (str): The name of the model to load the tokenizer for. Defaults to "NousResearch/Llama-2-7b-hf".

    Returns:
        AutoTokenizer: The loaded tokenizer.
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer

def load_model(model_name: str = "NousResearch/Llama-2-7b-hf", quantized = True) -> AutoModelForCausalLM:
    '''
    Load a model from the Hugging Face Hub.

    Args:
        model_name (str): The name of the model to load. Defaults to "NousResearch/Llama-2-7b-hf".
        quantized (bool): Whether to load the model in quantized format. Defaults to True.

    Returns:
        AutoModelForCausalLM: The loaded model.
    '''
    model = None
    if quantized:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_quant_type = "nf4",        # How we want to quantize our model / How our weights are stored (Normal Float 4-bit).
            bnb_4bit_compute_dtype = "float16"  # How we handle computations (16-bit float) -> higher precision for calculations.
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config = quantization_config,
            device_map = "auto",
            trust_remote_code = True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code = True,
            low_cpu_mem_usage = True  # Use low memory usage for CPU execution.
        )
        model.to("cpu")  # Move model to CPU if not quantized.

    model.config.use_cache = False              # Disable cache for training because it can lead to memory issues.
    model.config.gradient_checkpointing = True  # Enable gradient checkpointing to save memory during training.
    
    return model