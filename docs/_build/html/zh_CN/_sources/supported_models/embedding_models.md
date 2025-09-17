# Embedding Models

RTP-LLM supports the deployment of mainstream Embedding, Reranker, and Classifier models, with dedicated handling for multi-embedding architectures such as BGE-M3, enabling hybrid request processing within a single service instance. Built on Sentence Transformers, it allows users to tailor post-processing workflows to standard model architectures.

At the model layer, RTP-LLM leverages high-performance compute kernels to accelerate inference. The engine optimizes both intra- and inter-request sequence batching according to user configuration, eliminating redundant computation and improving GPU utilization.

## Example Launch Command

```shell
# TASK_TYPE in ["DENSE_EMBEDDING", "CLASSIFIER", "RERANKER", "BGE_M3"]
# For model with SentenceTransformer config, task_type can be auto deduced to DENSE EMBEDDING
/opt/conda310/bin/python3 -m rtp_llm.start_server \
--checkpoint_path /models/bert \
--model_type bert \
--act_type fp16 \
--start_port 8088 \
--TASK_TYPE DENSE_EMBEDDING \
--MAX_CONTEXT_BATCH_SIZE 20
```
## Example Client Request
### Dense Embedding
```python
import requests
url = "http://localhost:30000"
text_input = "Hello, what's your name"
request = {
    "input": ["text_input"]
}
response = requests.post(url + "/v1/embeddings", json=request).json()
```
### Reranker
```python
import requests
url = "http://localhost:30000"
request = {
    "query": "coffee",
    "documents": [
        "Starbuck",
        "Luckin",
        "Peets Coffee",
        "One point point"
    ]
}
response = requests.post(url + "/v1/reranker", json=request).json()
print(response)
```
### Classifier
``` python
import requests
url = "http://localhost:30000"
request = {
    "input": [
        [
            "what is panda?",
            "hi"
        ],
    ]
}
response = requests.post(url + "/v1/classifier", json=request).json()
print(response)
```

### BGE_M3
```python
import requests
url = "http://localhost:30000"
text_input = "Hello, what's your name"
request = {
    "input": ["text_input"]
}
endpoints = ["/v1/embeddings/dense", "/v1/embeddings/sparse", "/v1/embeddings/colbert"]
for endpoint in endpoints:
    response = requests.post(url + endpoint, json=request).json()
    print(response)
```

## Supported models

| Model Family (Embedding)                        | Example HuggingFace Identifier                | Chat Template | Description                                                                                                                          |
|-------------------------------------------------|-----------------------------------------------|---------------|--------------------------------------------------------------------------------------------------------------------------------------|
| **Qwen3 Embedding/Reranker**      | `Qwen/Qwen3-Embedding-8B`             | N/A           | Support all size of qwen3 embedding/reranker                   |
|
| **BGE (BgeEmbeddingModel)**                     | `BAAI/bge-large-en-v1.5`                        | N/A                 | only support BGE family with model_type=`Bert/Roberta/Qwen2`  including bge_m3, not suport `ModernBert` or `NewModel` . Specially, please set `model_type=qwen_2_embedding` for `Alibaba-NLP/gte-Qwen2-7B-instruct`  |
