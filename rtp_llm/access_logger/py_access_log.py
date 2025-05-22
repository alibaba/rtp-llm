from typing import Optional, Union, List, Dict, Any

import time
import json
import traceback

LoggableResponseTypes = Union[str, Dict[str, Any]]

class RequestLog:
    def __init__(self,
                 request_json: Optional[Dict[str, Any]] = None,
                 request_str: Optional[str] = None) -> None:
        self.request_json: Optional[Dict[str, Any]] = request_json
        self.request_str: Optional[str] = request_str

    @staticmethod
    def from_request(request: Union[Dict[str, Any], str]) -> 'RequestLog':
        if isinstance(request, dict):
            return RequestLog(request_json=request)
        elif isinstance(request, str):
            return RequestLog(request_str=request)
        else:
            raise Exception("unkown request type!")

class ResponseLog:
    def __init__(self) -> None:
        self.responses: List[LoggableResponseTypes] = []
        self.exception: Optional[BaseException] = None
        self.exception_traceback: Optional[str] = None

    def add_response(self, response: LoggableResponseTypes) -> None:
        self.responses.append(response)

    def add_exception(self, exception: BaseException) -> None:
        self.exception = exception
        self.exception_traceback = "\n".join(traceback.format_tb(exception.__traceback__))

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
            log_time: Optional[str] = None
    ):
        self.request = request
        self.response = response
        self.id = id
        if log_time is None:
            current_time = time.time()
            local_time = time.localtime(current_time)
            log_time = time.strftime("%Y-%m-%d %H:%M:%S", local_time) + f".{int((current_time % 1) * 1000):03d}"
        self.log_time = log_time

    @classmethod
    def from_json(cls, json_str: str):
        return cls(**json.loads(json_str))

