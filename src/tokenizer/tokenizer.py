import transformers
from typeguard import typechecked


@typechecked
def load_tokenizer(
    tokenizer_name: str,
) -> transformers.tokenization_utils_tokenizers.TokenizersBackend:
    # WARNING: This is a hacky version. Only tested on "FacebookAI/roberta-base"
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.eos_token_id == tokenizer.sep_token_id:
        tokenizer.add_special_tokens({"sep_token": "<sep>"})
    assert tokenizer.sep_token_id + 1 == len(tokenizer)
    return tokenizer
