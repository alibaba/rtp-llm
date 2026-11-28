"""Probability transport checks using real target verification and native MTP."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from request_smoke import require, validate_answer, validate_probabilities

COPY_TEXT = "a0 b1 c2 d3 e4 f5 g6 h7 i8 j9 k0 l1 m2 n3 o4 p5"


def check_logprobs(body, *, top_count, expected=COPY_TEXT, finish="stop"):
    message = validate_answer(body, expected=expected, finish=finish)
    rows = (body["choices"][0].get("logprobs") or {}).get("content") or []
    validate_probabilities(rows, top_count)
    pieces = []
    for row in rows:
        pieces.append(
            bytes(row["bytes"])
            if row.get("bytes") is not None
            else row["token"].encode("utf-8")
        )
    require(
        b"".join(pieces) == (message.get("content") or "").encode("utf-8"),
        "probability tokens do not reconstruct the returned content",
    )
    require(
        any(row["logprob"] < -1e-6 for row in rows),
        "compact logprobs unexpectedly became a greedy point mass",
    )


def run_logprobs_cases(client):
    payload = client.payload(f"原样复制以下字符串，只输出字符串本身：{COPY_TEXT}")
    # A greedy request still exposes the original target distribution.
    payload.update(temperature=1.0, top_k=1)
    for stream in (False, True):
        for top_count in (0, 3):
            request = {
                **payload,
                "stream": stream,
                "logprobs": True,
                "logprobs_mode": "original",
                "top_logprobs": top_count,
            }
            if stream:
                request["stream_options"] = {"include_usage": True}
            client.openai(
                f"logprobs_original_{'sse' if stream else 'sync'}_top{top_count}",
                request,
                lambda body, k=top_count: check_logprobs(body, top_count=k),
            )
    barrier = Barrier(3)

    def send(mode):
        request = {**payload, "logprobs": mode != "none"}
        if mode != "none":
            request.update(top_logprobs=3)
            if mode == "original":
                request["logprobs_mode"] = "original"

        def check(body):
            if mode == "none":
                validate_answer(body, COPY_TEXT)
                require(
                    not body["choices"][0].get("logprobs"),
                    "unrequested OpenAI probabilities",
                )
            else:
                check_logprobs(body, top_count=3)

        barrier.wait(timeout=30)
        return client.openai("logprobs_mixed_" + mode, request, check)

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(send, ("implicit", "original", "none")))
    for limit in (3, 4, 5):
        client.openai(
            f"logprobs_length_stop_{limit}",
            {**payload, "max_tokens": limit, "logprobs": True, "top_logprobs": 0},
            lambda body: check_logprobs(
                body, top_count=0, expected=None, finish="length"
            ),
        )
