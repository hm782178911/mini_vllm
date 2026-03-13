from transformers import AutoTokenizer


class TokenizerWrapper:

    def __init__(self, model_name):

        print("Loading tokenizer:", model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(self, text):

        encoded = self.tokenizer(text, return_tensors="pt")

        return encoded.input_ids

    def decode(self, tokens):

        return self.tokenizer.decode(tokens, skip_special_tokens=True)