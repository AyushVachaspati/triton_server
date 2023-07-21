import triton_python_backend_utils as pb_utils
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import traceback
import numpy as np

class TritonPythonModel:
    def initialize(self, args):
        print("Loading Model")
        model_path = "/models/santacoder_huggingface/assets/models/santacoder"
        checkpoint = "bigcode/santacoder"
        self.device = "cuda"  # "cuda" for GPU usage or "cpu" for CPU usage
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint,cache_dir=model_path)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.bos_token
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint,cache_dir=model_path,trust_remote_code=True,torch_dtype=torch.float16).to(self.device)
        print("Model Loaded")

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                in_text = pb_utils.get_input_tensor_by_name(request, "input")
                in_text = in_text.as_numpy()[0].decode('utf-8')
                
                tokens = self.tokenizer(in_text, return_tensors="pt").to(self.device)
                output = self.model.generate(**tokens,pad_token_id=self.tokenizer.eos_token_id,min_new_tokens=1,max_new_tokens=25)
                result = self.tokenizer.decode(output[0]).rstrip(self.tokenizer.convert_ids_to_tokens(self.tokenizer.eos_token_id))
                
                inference_response = pb_utils.InferenceResponse(output_tensors=[
                    pb_utils.Tensor("output",np.array([result],dtype=object))
                ])
                responses.append(inference_response)
            except:
                traceback.print_exc()
        
        return responses

    def finalize(self):
        print('Closing Server')