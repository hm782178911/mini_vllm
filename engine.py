import torch
from transformers import AutoModelForCausalLM


class InferenceEngine:

    def __init__(self, model_name):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("Loading model:", model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )

        self.model.eval()

    # -------------------------
    # Prefill (prompt阶段)
    # -------------------------
    def prefill(self, input_ids):

        with torch.no_grad():

            outputs = self.model(
                input_ids=input_ids,
                use_cache=True
            )

        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

        next_token = torch.argmax(logits, dim=-1)

        return next_token, past_key_values


    # -------------------------
    # Decode (逐token生成)
    # -------------------------
    def decode(self, token, past_key_values):

        with torch.no_grad():

            outputs = self.model(
                input_ids=token,
                past_key_values=past_key_values,
                use_cache=True
            )

        logits = outputs.logits[:, -1, :]

        next_token = torch.argmax(logits, dim=-1)

        past_key_values = outputs.past_key_values

        return next_token, past_key_values


    # -------------------------
    # Streaming生成
    # -------------------------
    def generate_stream(self, input_ids, max_new_tokens=50):

        input_ids = input_ids.to(self.device)

        # ---------- Prefill ----------
        next_token, past_key_values = self.prefill(input_ids)

        yield next_token.item()

        token = next_token.unsqueeze(-1)

        # ---------- Decode ----------
        for _ in range(max_new_tokens - 1):

            next_token, past_key_values = self.decode(token, past_key_values)

            yield next_token.item()

            token = next_token.unsqueeze(-1)