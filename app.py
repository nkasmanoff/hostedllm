#flask hosted app for LLM

from flask import Flask, request, jsonify
from llama2 import HuggingFaceLLM
from llama2_quantized import Llama7BQLLM

#llm_model = HuggingFaceLLM()
quantized_model = Llama7BQLLM()
app = Flask(__name__)


# set base route
@app.route('/')
def index():
    return {'message': 'Welcome to the LLM API'}



@app.route('/chat_llama', methods=['POST'])
def llm_prompt():
    data = request.get_json(force=True)
    prompt = data['inputs']
    llm_model = Llama7BQLLM()
    # This will be super slow if it loads the model each time, even with checkpointing
    response = llm_model(prompt=prompt)
    return jsonify(response)

if __name__ == '__main__':
    app.run()



