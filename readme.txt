How to build docker image
----------------------------
docker buildx build --platform linux/amd64 -f docker/Dockerfile -t koomingchin/anything-llm-qms:v1.15.0.2 .
docker save -o anything-llm-qms.tar koomingchin/anything-llm-qms:v1.15.0.2
docker load -i anything-llm-qms.tar


v1.15.0.2 (13-Oct-2026)
------------------------
- Fix empty reply when thinking is on for QWEN 3.5


How to allow Automatic Tool Call Approval to generate documents
----------------------------------------------------------------
In ../Documents/anythingllm/.env, add
AGENT_AUTO_APPROVED_SKILLS="<all>"