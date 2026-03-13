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

    def generate_stream(self, input_ids, max_new_tokens=50):

        input_ids = input_ids.to(self.device)

        past_key_values = None

        for _ in range(max_new_tokens):

            with torch.no_grad():

                outputs = self.model(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    use_cache=True
                )

            logits = outputs.logits[:, -1, :]

            past_key_values = outputs.past_key_values

            next_token = torch.argmax(logits, dim=-1)

            yield next_token.item()

            input_ids = next_token.unsqueeze(0)