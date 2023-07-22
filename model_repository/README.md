# Triton Server
Add the model files in the assets folder and change the model path in model.py file to reflect.

## Command to start Triton Docker Container
 docker run --gpus=all -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}:/workspace/ -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:23.06-py3 bash

## Command to start server within Docker Container
 tritonserver --model-repository=/models --log-verbose=1
 tritonserver --model-repository=/models --log-verbose=1 --cache-config local,size=1048576
## Curl to Query Triton Server
 curl.exe  -X POST  http://127.0.0.1:8000/v2/models/santacoder_huggingface/infer -H "Content-Type: application/json" -H "Accept: application/json" -d
 '{\"id\":\"test123\",\"inputs\":[{\"name\":\"input\", \"shape\":[1], \"datatype\": \"BYTES\", \"data\":[\"Complete this string\"]}]}'

 curl.exe -v 127.0.01:8000/v2/models/santacoder_huggingface
 curl.exe -v 127.0.01:8000/v2
 curl.exe -v 127.0.01:8000/v2/health/live
 curl.exe -v 127.0.01:8000/v2/health/ready