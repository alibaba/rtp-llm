curl --location 'http://127.0.0.1:7001/api/v2/services/aigc/text-generation/chat/completions' \
--header 'Authorization: Bearer M4TTTOGSZS' \
--header 'Content-Type: application/json' \
--header 'PreheatFlow: 1' \
--data '{
    "model": "PreheatFlowModel",
    "top_p": 0.1,
    "temperature": 1.0,
    "max_new_tokens": 1,
    "messages": [
        {
            "role": "user",
            "content": "你好!"
        }
    ]
}'