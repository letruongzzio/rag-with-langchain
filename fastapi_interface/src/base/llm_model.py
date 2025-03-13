import torch
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)


def get_hf_llm(model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
               max_new_token = 1024,
               **kwargs):
    """
    Get a Hugging Face language model pipeline

    Args:
        model_name (str): The Hugging Face model name
        max_new_token (int): The maximum number of tokens to generate
        kwargs: Additional keyword arguments

    Returns:
        pipeline: The Hugging Face pipeline
    """

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        low_cpu_mem_usage=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_token,
        pad_token_id=model.eos_token_id,
        device_map="auto"
    )

    llm = HuggingFacePipeline(
        pipeline=model_pipeline,
        model_kwargs=kwargs
    )

    return llm
