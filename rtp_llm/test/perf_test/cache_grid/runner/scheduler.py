"""Scheduler configuration with a common acknowledgement contract."""

import logging
import time


def configure_scheduler(
    port, *, batch_size, mode, post, timeout=60, attempts=1, retry_delay=3
):
    """Require an explicit successful acknowledgement before issuing requests."""
    if batch_size <= 0:
        raise ValueError("scheduler batch_size must be positive")
    if mode not in ("decode", "prefill"):
        raise ValueError(f"unsupported scheduler mode: {mode}")
    payload = {"batch_size": batch_size, "mode": mode}
    for attempt in range(1, attempts + 1):
        try:
            response = post(
                f"http://127.0.0.1:{port}/update_scheduler_info",
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            acknowledgement = response.json()
            if (
                not isinstance(acknowledgement, dict)
                or acknowledgement.get("status") != "ok"
                or acknowledgement.get("error")
            ):
                raise RuntimeError(f"scheduler rejected {payload}: {acknowledgement}")
            logging.info(
                "scheduler configured: payload=%s response=%s", payload, acknowledgement
            )
            return acknowledgement
        except Exception as error:
            if attempts == 1:
                raise
            if attempt == attempts:
                raise RuntimeError(
                    f"failed to configure scheduler after retries: {error!r}"
                ) from error
            logging.warning(
                "failed to configure scheduler, retrying (%d/%d): %r",
                attempt,
                attempts,
                error,
            )
            time.sleep(retry_delay)
