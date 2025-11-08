from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

class GemmaTranslator:
    def __init__(self, model_name: str = "google/gemma-3-27b-it", fallback_model: str = ""):
        self.model_name = model_name
        self.fallback_model = fallback_model
        self.device = 0 if torch.cuda.is_available() else -1
        self.tokenizer = None
        self.model = None
        self.pipe = None
        self._load_model()

    def _load_model(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.float16 if torch.cuda.is_available() else None,
            )
            self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=self.device)
        except Exception as e:
            print("Failed loading Gemma primary:", e)
            if self.fallback_model:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.fallback_model)
                    self.model = AutoModelForCausalLM.from_pretrained(self.fallback_model)
                    self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=self.device)
                    print("Loaded fallback model")
                except Exception as e2:
                    print("Failed loading fallback:", e2)
                    self.pipe = None
            else:
                self.pipe = None

    def translate(self, text: str, target_lang: str = "fa", max_new_tokens: int = 1024):
        if not self.pipe:
            return "Translation model not loaded on server."
        prompt = (
            f"You are a professional translator specialized in machine learning and engineering. "
            f"If there is code in the text, keep it unchanged. Translate the following text to {target_lang} preserving structure and meaning:\n\n"
            + text
            + "\n\nTranslation:"
        )
        out = self.pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)[0]["generated_text"]
        if "Translation:" in out:
            return out.split("Translation:",1)[1].strip()
        return out
