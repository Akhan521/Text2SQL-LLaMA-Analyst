from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers import GenerationConfig
import torch

def configure_generation(model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> GenerationConfig:
    '''
    Configures generation settings for the model.
    
    Args:
        model (PreTrainedModel): The model to configure for generation.
        tokenizer (PreTrainedTokenizer): The tokenizer associated with the model.

    Returns:
        GenerationConfig: The configured generation settings.
    '''
    generation_config = GenerationConfig.from_pretrained(model.name_or_path)

    generation_config.pad_token_id = tokenizer.eos_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id
    generation_config.max_new_tokens = 256
    generation_config.temperature = 0.7
    generation_config.top_p = 0.9
    generation_config.do_sample = True

    return generation_config

def generate(prompt: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, device: torch.device) -> str:
    '''
    Generate text based on a given prompt using the model.

    Args:
        prompt (str): The input prompt for text generation.
        model (PreTrainedModel): The model to use for generation.
        tokenizer (PreTrainedTokenizer): The tokenizer to encode the prompt.
        device (torch.device): The device to run the model on.

    Returns:
        str: The generated text.
    '''
    model.config.use_cache = True           # Enable cache for generation to speed up inference.
    model.gradient_checkpointing_disable()  # Disable gradient checkpointing for generation.
    model.to(device)
    model.eval()

    generation_config = configure_generation(model, tokenizer)
    encoded_prompt = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model.generate(
            input_ids = encoded_prompt,
            generation_config = generation_config,
            repetition_penalty = 2.0,
        )
    decoded_string = tokenizer.decode(output[0], clean_up_tokenization_spaces = True)
    return decoded_string
