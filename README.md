# Mini vLLM Demo

A minimal LLM inference service with:

- KV Cache decoding
- Streaming token generation
- FastAPI server

## Install

pip install -r requirements.txt

## Run

uvicorn server:app --host 0.0.0.0 --port 8000

## Test

curl -X POST http://localhost:8000/generate \
-H "Content-Type: application/json" \
-d '{"prompt":"Explain GPU"}'