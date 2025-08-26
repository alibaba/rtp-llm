import asyncio
import json
from typing import Any, AsyncGenerator, Dict

import aiohttp


async def post_async_chat_request(
    host: str = "127.0.0.1", port: int = 26000, request_data: Dict[str, Any] = None
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Sends an asynchronous POST request to the /v1/chat/completions endpoint.
    This function handles both streaming and non-streaming responses.

    Args:
        host (str): The hostname or IP address of the server.
        port (int): The port of the server.
        request_data (Dict): The payload for the chat completion request.

    Yields:
        Dict[str, Any]: A dictionary representing a chunk of the response.
    """
    url = f"http://{host}:{port}/v1/chat/completions"

    if request_data is None:
        request_data = {
            "messages": [{"role": "user", "content": "Hello, what's your name?"}],
            "model": "your-model-name",
            "stream": True,  # Set to False for a non-streaming response
        }

    try:
        # Use an aiohttp ClientSession for efficient connections
        async with aiohttp.ClientSession() as session:
            print(
                f"Sending async request to {url} with data: {json.dumps(request_data, indent=2)}"
            )

            async with session.post(url, json=request_data) as response:
                response.raise_for_status()  # Raise an exception for bad status codes

                if request_data.get("stream", False):
                    # Handle streaming response
                    async for chunk in response.content.iter_any():
                        try:
                            # The stream might contain multiple JSON objects
                            # We need to split them by newlines
                            for line in chunk.decode("utf-8").strip().split("\n"):
                                if line.startswith("data:"):
                                    line_data = line.split("data:")[1].strip()
                                    if line_data == "[DONE]":
                                        yield {"status": "done"}
                                        return
                                    yield json.loads(line_data)
                                else:
                                    print(f"Ignoring non-data line: {line}")
                        except (json.JSONDecodeError, UnicodeDecodeError) as e:
                            print(f"Error decoding chunk: {e}, chunk: {chunk}")

                else:
                    # Handle a standard, non-streaming response
                    full_response = await response.json()
                    yield full_response

    except aiohttp.ClientError as e:
        print(f"An aiohttp error occurred: {e}")
        yield {"error": str(e)}


async def main():
    # Example chat completion request payload
    # This assumes the model is configured on the backend
    chat_payload = {
        "messages": [{"role": "user", "content": "Tell me a short story."}],
        "model": "your-model-name",  # Replace with a valid model name
        "stream": True,  # Set to True for streaming, False for a single response
    }

    print("Starting chat request...")
    # The 'await' keyword pauses execution until the async generator is complete
    async for chunk in post_async_chat_request(request_data=chat_payload):
        print(chunk)
        # You can process each chunk here, e.g., display the text as it arrives


if __name__ == "__main__":
    try:
        # Run the main asynchronous function
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Client stopped.")
