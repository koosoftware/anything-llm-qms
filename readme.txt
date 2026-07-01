How to build docker image
----------------------------
docker buildx build --platform linux/amd64 -f docker/Dockerfile -t koomingchin/anything-llm-qms:v1.15.0 .
docker save -o anything-llm-qms.tar koomingchin/anything-llm-qms:v1.15.0
docker load -i anything-llm-qms.tar
docker images
docker login
docker push koomingchin/anything-llm-qms:v1.15.0