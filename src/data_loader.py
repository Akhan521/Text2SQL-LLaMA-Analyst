from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from typing import Tuple
import torch

def load_data(dataset_name: str = "ChrisHayduk/Llama-2-SQL-Dataset") -> Tuple[Dataset, Dataset]:
    '''
    Load a dataset from the Hugging Face Hub.

    Args:
        dataset_name (str): The name of the dataset to load. Defaults to "ChrisHayduk/Llama-2-SQL-Dataset".

    Returns:
        Tuple[Dataset, Dataset]: A tuple containing the training and evaluation datasets.
    '''
    dataset = load_dataset(dataset_name)
    train_dataset = dataset['train'].shuffle(seed = 42).select(range(1000)) # Shuffle and select first 1000 examples for training.
    eval_dataset = dataset['eval'].shuffle(seed = 42).select(range(100))    # Shuffle and select first 100 examples for evaluation.
    return train_dataset, eval_dataset

def construct_datapoint(example: dict, tokenizer: AutoTokenizer) -> torch.Tensor:
    '''
    Construct a single datapoint from a training example.
    
    Args:
        example (dict): A dictionary containing the example data ('input' and 'output').

    Returns:
        torch.Tensor: A tensor representation of the combined input and output.
    '''
    combined = example['input'] + example['output']
    return tokenizer(combined, padding = True, return_tensors = "pt")

def preprocess_dataset(dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    '''
    Preprocess the dataset by combining input and output fields and tokenizing them.

    Args:
        dataset (Dataset): The training dataset to preprocess.
        tokenizer (AutoTokenizer): The tokenizer to use for tokenization.

    Returns:
        Dataset: The preprocessed training dataset with tokenized inputs + outputs.
    '''
    return dataset.map(lambda example: construct_datapoint(example, tokenizer))
    