# Background

The framework currently provides multiple embedding formats:

* DENSE_EMBEDDING
* ALL_EMBEDDING
* SPARSE_EMBEDDING
* COLBERT_EMBEDDING

All accelerated through rtp_llm cpp framework.

# Dense Embedding

- For sentence transformer format checkpoints, supports computation according to sentence transformer prescribed processes
- For other types of checkpoints, normalizes transformer layer results and outputs

## Request and Response Format

Basically consistent with OpenAI format, see [OpenAI embedding](https://platform.openai.com/docs/api-reference/embeddings/create) and `rtp_llm/models/downstream_modules/embedding/api_datatype.py` for details.

# All Embedding

* Returns token IDs after prompt encoding, and returns embeddings for all tokens

# Reranker

Supports deployment of reranking models like bge-reranker-large, with request and response formats as follows. See `rtp_llm/models/downstream_modules/reranker/api_datatype.py` for details.

## Request

- query (str) - Query input
- documents (List[str]) - Documents to be ranked
- top_k (int, optional, defaults to None) - Number of documents to return
- truncation (bool, optional, defaults to True) - Whether to truncate oversized inputs
  -  If True, truncate to the longest input the model can accept
  -  If False, error for oversized inputs

## Response

- results (List[RerankingResult]) - List of RerankingResult in the following format:
  -  index (int) - Document position in input
  -  document (str) - Document content
  -  relevance_score (float) - Relevance score
- total_tokens (int) - Total number of tokens

# Usage

Specify `TASK_TYPE` information through environment variables, example:

``` python
### load model ###
# After entering the container, cd to FasterTransformer directory
# start http service
TASK_TYPE=ALL_EMBEDDING TOKENIZER_PATH=/path/to/tokenizer CHECKPOINT_PATH=/path/to/model MODEL_TYPE=your_model_type FT_SERVER_TEST=1 python3 -m rtp_llm.start_server
# request to server
curl -XPOST http://localhost:8088 -d '{"input": ["hello world", "my name is jack"], "model": "your_model_type"}'
```

TASK_TYPE Mapping

- Reranker model: TASK_TYPE=RERANKER
- Embedding model: TASK_TYPE=DENSE_EMBEDDING
- All_embedding model: TASK_TYPE=ALL_EMBEDDING

Note: Currently embedding models only support fp16 precision type
