import triton_python_backend_utils as pb_utils
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import torch
import os
import traceback
import numpy as np


# Output Streamer Implements the put() and end() methods from the BaseStreamer interface in HuggingFace Transformers libaray.
class OutputStreamer():
    def __init__(self, tokenizer: "AutoTokenizer", skip_prompt: bool = False, response_senders = [], eos_token = "<|end|>"):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.response_senders = response_senders
        self.next_tokens_are_prompt = True
        self.eos_token = eos_token
    
    def put(self, value):
        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return
        
        outputs = self.tokenizer.batch_decode(value)
        for i in range(len(outputs)):
            try:
                inference_response = pb_utils.InferenceResponse(output_tensors=[
                    pb_utils.Tensor("output",np.array([outputs[i]],dtype=object))
                ])
                if(outputs[i]==self.eos_token):
                    self.response_senders[i].send(inference_response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
                else:
                    self.response_senders[i].send(inference_response)
            except:
                # Trying to send response on closed connection
                pass
        
    def end(self):
        for response_sender in self.response_senders:
            try:
                response_sender.send(pb_utils.InferenceResponse(output_tensors=[
                        pb_utils.Tensor("output",np.array([self.eos_token],dtype=object))
                    ]),flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
            except:
                # Trying to send response on closed connection
                pass


class TritonPythonModel:

    def addSystemPrompt(self,in_text):
        with open("/models/starcoder_chat/assets/prompt.txt","r") as f:
            systemPrompt = f.read()
        print(systemPrompt + in_text)
        return systemPrompt + in_text
    

    def initialize(self, args):
        print("Loading Model StarCoder Chat Beta")
        model_path = "/models/starcoder_chat/assets/models/starcoder_chat"
        login("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")
        checkpoint = "HuggingFaceH4/starchat-beta"
        self.device = "cuda:1"  # "cuda" for GPU usage or "cpu" for CPU usage
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint,cache_dir=model_path)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.bos_token
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint,cache_dir=model_path,trust_remote_code=True,torch_dtype=torch.float16).to(self.device)
        print("Model Loaded")

    def execute(self, requests):
        response_senders = []
        inputs = []

        for request in requests:
            in_text = pb_utils.get_input_tensor_by_name(request, "input")
            in_text = in_text.as_numpy()[0][0].decode('utf-8')
            in_text = self.addSystemPrompt(in_text);
            inputs.append(in_text)
            response_senders.append(request.get_response_sender())

        streamer = OutputStreamer(self.tokenizer,skip_prompt=True,response_senders=response_senders)
        inputs = self.tokenizer(inputs,padding=True,return_tensors="pt").to(self.device)
        # We use a special <|end|> token with ID 49155 to denote ends of a turn ( "<|end|>" )
        self.model.generate(**inputs,streamer=streamer,
                            min_new_tokens=0,max_new_tokens=500,
                            pad_token_id=49155,
                            eos_token_id=49155)
        
        return None
        
    def finalize(self):
        print('Closing Server')