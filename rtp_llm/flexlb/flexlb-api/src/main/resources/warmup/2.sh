curl --location 'http://127.0.0.1:7001/api/v1/services/aigc/text-generation/generation' \
--header 'Authorization: Bearer M4TTTOGSZS' \
--header 'Content-Type: application/json' \
--header 'PreheatFlow: 1' \
--data '{
    "model": "PreheatFlowModel",
    "generate_config": {
        "temperature": 1,
        "max_new_tokens": 1
    },
    "prompt": "Human: 你好！\n\nAssistant: "
}'