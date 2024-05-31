# 背景
* 框架目前提供了多种embedding格式：DENSE_EMBEDDING/ALL_EMBEDDING/SPARSE_EMBEDDING/COLBERT_EMBEDDING 通过rtp_llm cpp框架进行加速

# dense embedding
- 对于sentence transformer格式的checkpoint，支持以sentence transformer规定的流程进行计算
- 对于其他类型的checkpoint，将transformer层的结果进行normalize后输出
## 请求和响应格式
与openai格式基本一致，详情参考[openai embedding](https://platform.openai.com/docs/api-reference/embeddings/create)和`maga_transformer/models/downstream_modules/embedding/api_datatype.py` 
# all embedding
* 返回prompt encode之后的token ids，并且返回所有token的embedding。

# reranker
支持部署bge-reranker-large等重排模型，请求和响应格式如下，具体可参考:maga_transformer/models/downstream_modules/reranker/api_datatype.py
## 请求
- query (str) - 请求
- documents (List[str]) - 需要排序的doc
- top_k (int, optional, defaults to None) - 返回文档数
- truncation (bool, optional, defaults to True) - 是否需要截断超长输入
  -  If True, 截断到模型能接受的最长输入
  -  If False, 超长报错
## 响应
- results (List[RerankingResult]) - RerankingResult列表，格式如下
  -  index (int) - 文档在输入中的位置
  -  document (str) - 文档内容
  -  relevance_score (float) - 相关性分数
- total_tokens (int) - 总token数

## 使用方法
通过环境变量指定TASK_TYPE信息,示例：
``` python
### load model ###
# 进入容器之后, cd 到 FasterTransformer目录
# start http service
TASK_TYPE=ALL_EMBEDDING  TOKENIZER_PATH=/path/to/tokenizer CHECKPOINT_PATH=/path/to/model MODEL_TYPE=your_model_type FT_SERVER_TEST=1 python3 -m maga_transformer.start_server
# request to server
curl -XPOST http://localhost:8088 -d '{"input": ["hello world", "my name is jack"], "model": "your_model_type"}'
```
TASK_TYPE映射
- reranker模型：TASK_TYPE=RERANKER
- embedding模型: TASK_TYPE=DENSE_EMBEDDING
- all_embedding模型: TASK_TYPE=ALL_EMBEDDING

注意：目前embeding模型仅支持fp16类型精度