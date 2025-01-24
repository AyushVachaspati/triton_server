# Triton Server
Add the model files in the assets folder and change the model path in model.py file to reflect.

## Command to start Triton Docker Container
 cd < triton_server folder >
 
 docker run --gpus=all -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}:/workspace/ -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:23.06-py3 bash
 
 pip install torch transformers

## Command to start server within Docker Container
 tritonserver --model-repository=/models --log-verbose=1
 tritonserver --model-repository=/models --log-verbose=1 --cache-config local,size=1048576
 tritonserver --model-repository=/models --log-verbose=1 --model-control-mode=explicit --load-model=santacoder_huggingface --cache-config local,size=1048576
pip install torch transformers; tritonserver --model-repository=/models --log-verbose=1 --model-control-mode=explicit --load-model=santacoder_huggingface --cache-config local,size=1048576

## Curl to Query Triton Server without Nginx
 curl.exe  -X POST  http://127.0.0.1:8000/v2/models/santacoder_huggingface/infer -H "Content-Type: application/json" -H "Accept: application/json" -d '{\"id\":\"test123\",\"inputs\":[{\"name\":\"input\", \"shape\":[1,1], \"datatype\": \"BYTES\", \"data\":[[\"Complete this string\"]]}]}'

 ### server Prometheus metrics
 http://127.0.0.1:8002/metrics

 curl.exe -v 127.0.01:8000/v2/models/santacoder_huggingface
 curl.exe -v 127.0.01:8000/v2/models/santacoder_huggingface/stats
 curl.exe -v 127.0.01:8000/v2
 curl.exe -v 127.0.01:8000/v2/health/live
 curl.exe -v 127.0.01:8000/v2/health/ready

## Starting NGINX
 C:\nginx\nginx.exe -c C:\nginx\conf\nginx.conf -p C:\nginx\

## Curl to Query Triton Server with Nginx Redirecting from port 80 to port 8000
 curl.exe  -X POST  http://127.0.0.1:80/v2/models/santacoder_huggingface/infer -H "Content-Type: application/json" -H "Accept: application/json" -d
 '{\"id\":\"test123\",\"inputs\":[{\"name\":\"input\", \"shape\":[1], \"datatype\": \"BYTES\", \"data\":[\"Complete this string\"]}]}'

 curl.exe -v 127.0.01:80/v2/models/santacoder_huggingface
 curl.exe -v 127.0.01:80/v2/models/santacoder_huggingface/stats
 curl.exe -v 127.0.01:80/v2
 curl.exe -v 127.0.01:80/v2/health/live
 curl.exe -v 127.0.01:80/v2/health/ready
 

 ## cloud instance startup script
pip install torch transformers;
git clone https://github.com/AyushVachaspati/triton_server.git /triton_server ;
tritonserver --model-repository=/triton_server/model_repository --model-control-mode=explicit --load-model=santacoder_huggingface_stream;
tritonserver --model-repository=/triton_server/model_repository --model-control-mode=explicit --load-model=santacoder_huggingface --cache-config local,size=1048576;


 ## command to load on the cloud
tritonserver --model-repository=/triton_server/model_repository --model-control-mode=explicit --load-model=starcoder_chat --load-model=starcoder_huggingface  --cache-config local,size=1048576 --model-load-thread-count=2
