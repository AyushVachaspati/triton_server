import triton_python_backend_utils as pb_utils
from transformers import AutoModelForCausalLM, AutoTokenizer,  TextIteratorStreamer
import torch
import os
import traceback
import numpy as np
import time
from threading import Thread

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
        response_sender = []
        inputs = []
        for request in requests:
            in_text = pb_utils.get_input_tensor_by_name(request, "input")
            in_text = in_text.as_numpy()[0].decode('utf-8')
            streamer = TextIteratorStreamer(self.tokenizer,skip_prompt=True)
            inputs = self.tokenizer([in_text],return_tensors="pt").to(self.device)
            # Run the generation in a separate thread, so that we can fetch the generated text in a non-blocking way.
            generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=256)
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            response_sender = request.get_response_sender()
            for new_text in streamer:
                inference_response = pb_utils.InferenceResponse(output_tensors=[
                    pb_utils.Tensor("output",np.array([new_text],dtype=object))
                ])
                response_sender.send(inference_response)
            response_sender.send(pb_utils.InferenceResponse(output_tensors=[
                    pb_utils.Tensor("output",np.array(["<|Endoftext|>"],dtype=object))
                ]),flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
        return None
    
        # return responses
    def finalize(self):
        print('Closing Server')