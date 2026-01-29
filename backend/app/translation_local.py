from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch

_MODEL = None
_TOKENIZER = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _ensure_model(model_name: str = "facebook/m2m100_418M"):
    global _MODEL, _TOKENIZER
    if _MODEL is None or _TOKENIZER is None:
        _TOKENIZER = M2M100Tokenizer.from_pretrained(model_name)
        _MODEL = M2M100ForConditionalGeneration.from_pretrained(model_name).to(_DEVICE)
    return _MODEL, _TOKENIZER

def translate_text_local(text: str, tgt_lang: str = "my", src_lang: str = "en", model_name: str = "facebook/m2m100_418M") -> str:
    model, tokenizer = _ensure_model(model_name)
    try:
        tokenizer.src_lang = src_lang
    except Exception:
        pass

    inputs = tokenizer(text, return_tensors="pt", padding=True).to(_DEVICE)
    forced_bos_token_id = tokenizer.get_lang_id(tgt_lang)
    with torch.no_grad():
        generated = model.generate(**inputs, forced_bos_token_id=forced_bos_token_id, max_length=512)
    out = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    return out
