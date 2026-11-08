"""Multilingual translation with NLLB-200 (distilled 600M)."""
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# NLLB language codes
LANGUAGES = {
    "Tamil": "tam_Taml",
    "Hindi": "hin_Deva",
    "French": "fra_Latn",
    "Spanish": "spa_Latn",
    "German": "deu_Latn",
}


class Translator:
    def __init__(self, model_name="facebook/nllb-200-distilled-600M", device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device

    def translate(self, text, target_codes):
        """Translate English text into each target language. Returns {code: translation}."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        results = {}
        for code in target_codes:
            forced_bos = self.tokenizer.convert_tokens_to_ids(code)
            out = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos,
                max_new_tokens=64,
            )
            results[code] = self.tokenizer.batch_decode(out, skip_special_tokens=True)[0]
        return results