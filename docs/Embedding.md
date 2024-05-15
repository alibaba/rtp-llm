# 背景
* 框架目前提供了多种embedding格式：DENSE_EMBEDDING/ALL_EMBEDDING/SPARSE_EMBEDDING/COLBERT_EMBEDDING

# dense embedding
* 在普通情况下：返回第一个或者或者最后一个token的embedding。
* 在SentenceTransformer情况下：(待完善)。

# all embedding
* 返回prompt encode之后的token ids，并且返回所有token的embedding。

## 使用方法
* 通过环境变量指定TASK_TYPE信息,示例
``` python
### load model ###
# 进入容器之后, cd 到 FasterTransformer目录
# start http service
TASK_TYPE=ALL_EMBEDDING  TOKENIZER_PATH=/path/to/tokenizer CHECKPOINT_PATH=/path/to/model MODEL_TYPE=your_model_type FT_SERVER_TEST=1 python3 -m maga_transformer.start_server
# request to server
curl -XPOST http://localhost:8088 -d '{"input": ["hello world", "my name is jack"], "model": "your_model_type"}'
```
* 请求和响应和[openai embedding](https://platform.openai.com/docs/api-reference/embeddings/create)格式一致。