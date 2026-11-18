from http import HTTPStatus

from vllm.entrypoints.openai.engine.protocol import GenerationError

RETRYABLE_ERROR_PREFIX = "navi_sched_retryable:"


class NaviSchedGenerationError(GenerationError):
    def __init__(self, message):
        super().__init__(message)
        self.status_code = HTTPStatus.TOO_MANY_REQUESTS


def raise_if_error(finish_reason, request_id, stop_reason, logger):
    if finish_reason != "error":
        return
    error = GenerationError("Internal server error")
    if isinstance(stop_reason, str) and stop_reason.startswith(RETRYABLE_ERROR_PREFIX):
        message = (
            stop_reason.removeprefix(RETRYABLE_ERROR_PREFIX)
            or "NaviSched remote request failed"
        )
        error = NaviSchedGenerationError(message)
    logger.error("Request %s failed during generation: %s", request_id, str(error))
    raise error
