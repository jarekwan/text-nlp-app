from functools import lru_cache

from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline


BERT_MODEL_NAME = "dkleczek/bert-base-polish-cased-v1"
MASK_TOKEN = "[MASK]"


@lru_cache(maxsize=1)
def get_fill_mask_pipeline():
    """Ładuje polski BERT fill-mask raz na instancję aplikacji."""
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(BERT_MODEL_NAME)
    return pipeline("fill-mask", model=model, tokenizer=tokenizer, device=-1)
