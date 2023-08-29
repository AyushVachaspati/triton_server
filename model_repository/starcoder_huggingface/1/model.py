import triton_python_backend_utils as pb_utils
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import torch
import os
import traceback
import numpy as np

class TritonPythonModel:
    def initialize(self, args):
        print("Loading Model StarCoder Plus")
        model_path = "/models/starcoder_huggingface/assets/models/starcoder"
        login("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")
        checkpoint = "bigcode/starcoderplus"
        self.device = "cuda:0"  # "cuda" for GPU usage or "cpu" for CPU usage
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint,cache_dir=model_path)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.bos_token
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint,cache_dir=model_path,trust_remote_code=True,torch_dtype=torch.float16).to(self.device)
        print("Model Loaded")

    def execute(self, requests):
        responses = []
        inputs = []
        eos_token = self.tokenizer.convert_ids_to_tokens(self.tokenizer.eos_token_id)
        try:
            for request in requests:
                in_text = pb_utils.get_input_tensor_by_name(request, "input")
                in_text = in_text.as_numpy()[0][0].decode('utf-8')
                inputs.append(in_text)
            
            tokens = self.tokenizer(inputs, padding=True, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**tokens,pad_token_id=self.tokenizer.eos_token_id,
                                          min_new_tokens=0,max_new_tokens=50)
            results = self.tokenizer.batch_decode(outputs)
            
            ## Removing Response for Empty Input Strings
            for i in range(len(results)):
                final_result = results[i].rstrip(eos_token)
                inference_response = pb_utils.InferenceResponse(output_tensors=[
                    pb_utils.Tensor("output",np.array([final_result],dtype=object))
                ])
                responses.append(inference_response)
        except:
            traceback.print_exc()
        
        return responses

    def finalize(self):
        print('Closing Server')