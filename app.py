from qllama2 import Llama2QLLM
from fastapi import FastAPI
import uvicorn
from flask import jsonify
import torch
#llm_model = HuggingFaceLLM()
quantized_model = Llama2QLLM()

base_url = 'http://localhost:8000'

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Welcome to the floodbrain llama2 api on FASTAPI"}

@app.post("/chat")
async def request_chat(prompt:str ):
    torch.cuda.empty_cache()
    response = quantized_model(prompt=prompt)
    return response



if __name__ == '__main__':
    uvicorn.run(app)
