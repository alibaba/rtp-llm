import json

import requests


def get_worker_status(host: str = "127.0.0.1", port: int = 26001, data: dict = None):
    """
    Sends a POST request to the /worker_status endpoint and returns the response.

    Args:
        host (str): The hostname or IP address of the server.
        port (int): The port of the server.
        data (dict): Optional dictionary to send in the request body.

    Returns:
        dict: The JSON response from the server.
    """
    url = f"http://{host}:{port}/worker_status"

    try:
        print(f"Sending request to {url} with data: {data}")
        response = requests.post(url, json=data)

        # Raise an exception for bad status codes (e.g., 404, 500)
        response.raise_for_status()

        # Parse and return the JSON response
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return {"error": str(e)}


# --- Example Usage ---
if __name__ == "__main__":
    # Assuming your server is running on localhost at port 8000
    server_port = 26000

    # You can send an empty dictionary or pass any required query parameters
    request_data = {}

    status_info = get_worker_status(port=server_port, data=request_data)

    if "error" in status_info:
        print("Failed to retrieve worker status.")
    else:
        print("Successfully retrieved worker status:")
        print(json.dumps(status_info, indent=4))
