"""Run concurrent requests after the K3 protocol smoke cases."""

from concurrent.futures import ThreadPoolExecutor

from request_smoke import validate_answer


def run_tail_health(client):
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(
                client.openai, f"tail_reuse_slot_{i}", client.payload(), validate_answer
            )
            for i in range(4)
        ]
        for future in futures:
            future.result()
