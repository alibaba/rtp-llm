# 使用方法

rtp-llm 支持 openai chat 格式的接口调用，可以作为 openai 接口服务的无缝 drop-in 。

启动
```bash
export TOKENIZER_PATH=/path/to/tokenizer
export CHECKPOINT_PATH=/path/to/model
export FT_SERVER_TEST=1
python3 -m maga_transformer.start_server
```

在 server 启动后，我们提供了一个 openai client 兼容的 chat 接口。使用示例如下：

``` python
import openai # you want `pip install openai==1.3.9`
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionUserMessageParam
openai.base_url = f"http://127.0.0.1:{int(os.environ['START_PORT'])}/"
openai.api_key = "none"

typed_messages: List[ChatCompletionMessageParam] = [
    ChatCompletionUserMessageParam(content="你是谁", role="user")
]

response1 = openai.chat.completions.create(
    model="whatever",
    messages=typed_messages
)
print(f"response: {response1}")

typed_messages.append(response1.choices[0].message)
typed_messages.append(ChatCompletionUserMessageParam(content="torch中的tensor和tensorflow有什么不同？", role="user"))

response2 = openai.chat.completions.create(
    model="whatever",
    messages=typed_messages,
    stream=True
)
for res in response2:
    print(f"response: {res.model_dump_json(indent=4)}")

```

或者，也可以使用 curl 调用，示例如下:

```python
chat_messages = [
    {
        "role": "user",
        "content": "你是谁"
    }
]
openai_request = {
    "temperature": 0.0,
    "top_p": 0.1,
    "messages": chat_messages
}
response = requests.post(f"http://localhost:{port}/v1/chat/completions", json=openai_request)
print(response.json())
```

# 配置 chat template

对于使用了 transformers tokenizer 的模型， 服务会读取 tokenizer_config.json 里的 chat_template ，如果 config 里带有这个字段，则会按照这个模板渲染。 chat_template 相关文档详见 https://huggingface.co/docs/transformers/chat_templating 。

# function call

openai 接口可以实现 function 调用。使用例子：
```python
function_request = {
    "model": "123",
    "top_p": 0.1,
    "temperature": 0.0,
    "messages": [
        {
            "role": "user",
            "content": "杭州市余杭区天气如何？",
        }
    ],
    "functions": [
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        }
    ]
}
response = requests.post(f"http://localhost:{port}/v1/chat/completions", json=function_request).json()
print(f"function response: {json.dumps(response, indent=4, ensure_ascii=False)}")

function_request["messages"].append(response["choices"][0]["message"])
function_request["messages"].append({
    "role": "function",
    "name": "get_current_weather",
    "content": '{"temperature": "22", "unit": "celsius", "description": "Sunny"}',
})
response = requests.post(f"http://localhost:{port}/v1/chat/completions", json=function_request).json()
print(f"final response: {json.dumps(response, indent=4, ensure_ascii=False)}")

function_request = {
    "model": "123",
    "top_p": 0.1,
    "temperature": 0.0,
    "messages": [
        {
            "role": "user",
            "content": "请帮我生成一张杭州西站的图片。",
        }
    ],
    "functions": [
        {
            "name": "google_search",
            "description": "谷歌搜索是一个通用搜索引擎，可用于访问互联网、查询百科知识、了解时事新闻等。 Format the arguments as a JSON object.",
            "parameters": {
                "type": "object",
                "properties": [{
                    "name": "search_query",
                    "description": "搜索关键词或短语",
                    "required": True,
                    "schema": {"type": "string"},
                }]
            },
        },
        {
            "name": "image_gen",
            "description": "image_gen 是一个AI绘画（图像生成）服务，输入文本描述，返回根据文本作画得到的图片的URL。Format the arguments as a JSON object.",
            "parameters": {
                "type": "object",
                "properties": [{
                    "name": "prompt",
                    "description": "英文关键词，描述了希望图像具有什么内容",
                    "required": True,
                    "schema": {"type": "string"},
                }]
            },
        }
    ]
}
response = requests.post(f"http://localhost:{port}/v1/chat/completions", json=function_request).json()
print(f"2 function response: {json.dumps(response, indent=4, ensure_ascii=False)}")
```

当前 function call 功能当前仅支持 qwen 系列模型，其他模型待扩展。
