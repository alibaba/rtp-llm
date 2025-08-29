import requests


class RTPHttpClient:
    def __init__(self, server_address: str):
        self._server_address: str = server_address

    def pause(self):
        url = f"http://{self._server_address}:26006/pause"
        try:
            response = requests.post(url)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")

    def restart(self):
        url = f"http://{self._server_address}:26006/restart"
        try:
            response = requests.post(url)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    # 示例用法
    client = RTPHttpClient("localhost")  # 替换为实际的服务器地址

    # 测试 pause
    client.pause()

    _ = input("任意键继续...")

    # 测试 restart
    client.restart()
