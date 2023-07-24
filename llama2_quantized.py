import transformers
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM


class Llama7BQLLM():
    def __init__(self, verbose=False):
        self.model_id = "TheBloke/Llama-2-7b-Chat-GPTQ"
        self.model_basename = "gptq_model-4bit-128g"
        self.verbose = verbose
        self.pipeline, self.bare_model, self.tokenizer = self.load_pipeline()
        self.chat_template = """
            Below is a conversation between a curious user named Noah and a helpful AI assistant named FloodBotAI.
            Noah and FloodBotAI are really into using AI to assist in flood prevention, disaster recovery, and rescue operations
            using Earth Observation data, so they form a Batman and Robin like duo. FloodBotAI is an AI LLM model that has been trained
            on a large corpus of flood-related data and is able to answer questions in a conversational manner, prioritizing accuracy
            as its recommendations have life and death consequences.

            Question: {query_str}
            Response: """

        self.blank_template = """
            {query_str}
            """

    def __call__(self, prompt):
        prompt = self.blank_template.format(query_str=prompt)
        response = self.pipeline(prompt)

        return response

    def load_pipeline(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        bare_model = AutoGPTQForCausalLM.from_quantized(
            self.model_id,
            model_basename=self.model_basename,
            use_safetensors=True,
            trust_remote_code=True,
            device="cuda:0",
            use_triton=False,
            quantize_config=None,

        )

        pipeline = transformers.pipeline(
            "text-generation",
            model=bare_model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.0,
            top_p=0.95,
            repetition_penalty=1.15
        )

        pipeline = HuggingFacePipeline(pipeline=pipeline, verbose=self.verbose)
        return pipeline, bare_model, tokenizer


if __name__ == '__main__':
    llm_model = Llama7BQLLM(verbose=True)

    # example function call would be

    response = llm_model("Any weather events in the next 24 hours you think we should be aware of? For example, do any cities look like they might be in risk of flooding?")