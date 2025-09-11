# Usage Guide

RTP-LLM supports OpenAI chat format API calls and can be used as a seamless drop-in replacement for OpenAI interface services.

## Getting Started

```bash
export TOKENIZER_PATH=/path/to/tokenizer
export CHECKPOINT_PATH=/path/to/model
export FT_SERVER_TEST=1
python3 -m rtp_llm.start_server
```

After the server starts, we provide an OpenAI client compatible chat interface. Usage examples are as follows:

``` python
import openai # you want `pip install openai==1.3.9`
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionUserMessageParam
openai.base_url = f"http://localhost:{int(os.environ['START_PORT'])}/"
openai.api_key = "none"

typed_messages: List[ChatCompletionMessageParam] = [
    ChatCompletionUserMessageParam(content="Who are you?", role="user")
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

use curl to get result, Usage example:

```python
chat_messages = [
    {
        "role": "user",
        "content": "Who are you?"
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

## Chat Template Configuration

For models using the transformers tokenizer, the service will read the chat_template from tokenizer_config.json. If this field is present in the config, it will be rendered according to this template. See https://huggingface.co/docs/transformers/chat_templating for chat_template documentation.

## Function Call

The OpenAI interface can implement function calling. Usage example:
```python
function_request = {
    "model": "123",
    "top_p": 0.1,
    "temperature": 0.0,
    "messages": [
        {
            "role": "user",
            "content": "What's the weather like in Yuhang District, Hangzhou?",
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
            "content": "Please generate an image of Hangzhou West Station.",
        }
    ],
    "functions": [
        {
            "name": "google_search",
            "description": "Google search is a general-purpose search engine that can be used to access the internet, query encyclopedic knowledge, understand current news, etc. Format the arguments as a JSON object.",
            "parameters": {
                "type": "object",
                "properties": [{
                    "name": "search_query",
                    "description": "Search keywords or phrases",
                    "required": True,
                    "schema": {"type": "string"},
                }]
            },
        },
        {
            "name": "image_gen",
            "description": "image_gen is an AI painting (image generation) service. Input text description and return the URL of the generated image based on text painting. Format the arguments as a JSON object.",
            "parameters": {
                "type": "object",
                "properties": [{
                    "name": "prompt",
                    "description": "English keywords describing the desired image content",
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

Currently, the function call feature only supports qwen series models, with support for other models coming soon.
