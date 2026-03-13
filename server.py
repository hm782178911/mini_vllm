from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

from engine import InferenceEngine
from tokenizer_wrapper import TokenizerWrapper


MODEL_NAME = "/home/hzy/project/qwen3_06B"

app = FastAPI()

engine = InferenceEngine(MODEL_NAME)
tokenizer = TokenizerWrapper(MODEL_NAME)


class GenerateRequest(BaseModel):
    prompt: str


def stream_generator(prompt):

    input_ids = tokenizer.encode(prompt)

    generated_tokens = []

    for token in engine.generate_stream(input_ids):

        # generated_tokens.append(token)

        # text = tokenizer.decode(generated_tokens)
        
        # yield text + "\n"
        token=tokenizer.decode([token])
        yield token
    
    yield '\n'

@app.post("/generate")
async def generate(req: GenerateRequest):

    return StreamingResponse(
        stream_generator(req.prompt),
        media_type="text/plain"
    )