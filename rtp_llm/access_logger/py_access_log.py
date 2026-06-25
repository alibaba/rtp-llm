import json
import time
import traceback
from typing import Any, Dict, List, Optional, Union

LoggableResponseTypes = Union[str, Dict[str, Any]]
LoggableInputIds = Union[List[int], List[List[int]]]
LoggableInputTokenLen = Union[int, List[int]]


class RequestLog:
    def __init__(
        self,
        request_json: Optional[Dict[str, Any]] = None,
        request_str: Optional[str] = None,
        input_ids: Optional[List[List[int]]] = None,
    ) -> None:
        self.request_json: Optional[Dict[str, Any]] = request_json
        self.request_str: Optional[str] = request_str
        self.input_ids: Optional[LoggableInputIds] = self._get_loggable_input_ids(
            input_ids
        )
        self.input_token_len: Optional[LoggableInputTokenLen] = (
            self._get_input_token_len(input_ids)
        )

    @staticmethod
    def _get_loggable_input_ids(
        input_ids: Optional[List[List[int]]],
    ) -> Optional[LoggableInputIds]:
        if input_ids is None:
            return None
        if len(input_ids) == 1:
            return input_ids[0]
        return input_ids

    @staticmethod
    def _get_input_token_len(
        input_ids: Optional[List[List[int]]],
    ) -> Optional[LoggableInputTokenLen]:
        if input_ids is None:
            return None
        input_token_lens = [len(input_id) for input_id in input_ids]
        if len(input_token_lens) == 1:
            return input_token_lens[0]
        return input_token_lens

    @staticmethod
    def from_request(
        request: Union[Dict[str, Any], str],
        input_ids: Optional[List[List[int]]] = None,
    ) -> "RequestLog":
        if isinstance(request, dict):
            return RequestLog(request_json=request, input_ids=input_ids)
        elif isinstance(request, str):
            return RequestLog(request_str=request, input_ids=input_ids)
        else:
            raise Exception("unkown request type!")


class ResponseLog:
    def __init__(self, output_ids: Optional[List[List[int]]] = None) -> None:
        self.responses: List[LoggableResponseTypes] = []
        self.output_ids: Optional[List[List[int]]] = output_ids
        self.exception: Optional[BaseException] = None
        self.exception_traceback: Optional[str] = None

    def add_response(self, response: LoggableResponseTypes) -> None:
        self.responses.append(response)

    def add_exception(self, exception: BaseException) -> None:
        self.exception = exception
        self.exception_traceback = "\n".join(
            traceback.format_tb(exception.__traceback__)
        )


class PyAccessLog:
    request: RequestLog
    response: ResponseLog
    id: int
    log_time: str

    def __init__(
        self,
        request: RequestLog,
        response: ResponseLog,
        id: int,
        log_time: Optional[str] = None,
    ):
        self.request = request
        self.response = response
        self.id = id
        self.request_id = id
        if log_time is None:
            current_time = time.time()
            local_time = time.localtime(current_time)
            log_time = (
                time.strftime("%Y-%m-%d %H:%M:%S", local_time)
                + f".{int((current_time % 1) * 1000):03d}"
            )
        self.log_time = log_time

    @classmethod
    def from_json(cls, json_str: str):
        return cls(**json.loads(json_str))
