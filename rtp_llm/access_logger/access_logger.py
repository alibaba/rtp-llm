import logging
from typing import Any, Dict

from rtp_llm.access_logger.json_util import dump_json
from rtp_llm.access_logger.log_utils import get_handler
from rtp_llm.access_logger.py_access_log import PyAccessLog, RequestLog, ResponseLog
from rtp_llm.config.py_config_modules import StaticConfig
from rtp_llm.structure.request_extractor import request_id_field_name

ACCESS_LOGGER_NAME = "access_logger"
QUERY_ACCESS_LOGGER_NAME = "query_access_logger"


def init_access_logger() -> None:
    access_logger = logging.getLogger(ACCESS_LOGGER_NAME)
    handler = get_handler("access.log")
    formatter = logging.Formatter("%(message)s")
    access_logger.handlers.clear()
    access_logger.parent = None
    if handler != None:
        handler.setFormatter(formatter)
        access_logger.addHandler(handler)


def init_query_access_logger() -> None:
    access_logger = logging.getLogger(QUERY_ACCESS_LOGGER_NAME)
    handler = get_handler("query_access.log")
    formatter = logging.Formatter("%(message)s")
    access_logger.handlers.clear()
    access_logger.parent = None
    if handler != None:
        handler.setFormatter(formatter)
        access_logger.addHandler(handler)


class AccessLogger:
    def __init__(self) -> None:
        init_access_logger()
        init_query_access_logger()
        self.logger = logging.getLogger(ACCESS_LOGGER_NAME)
        self.query_logger = logging.getLogger(QUERY_ACCESS_LOGGER_NAME)

    @staticmethod
    def is_private_request(request: Dict[str, Any]):
        return request.get(
            "private_request", StaticConfig.misc_config.disable_access_log
        )

    def log_access(self, request: Dict[str, Any], response: ResponseLog) -> None:
        request_log = RequestLog.from_request(request)
        access_log = PyAccessLog(
            request=request_log, response=response, id=request[request_id_field_name]
        )
        self.logger.info(dump_json(access_log))

    def log_query_access(self, request: Dict[str, Any]) -> None:
        if not self.is_private_request(request):
            request_log = RequestLog.from_request(request)
            response_log = ResponseLog()
            access_log = PyAccessLog(
                request=request_log,
                response=response_log,
                id=request[request_id_field_name],
            )
            self.query_logger.info(dump_json(access_log))

    def log_success_access(self, request: Dict[str, Any], response: Any) -> None:
        if not self.is_private_request(request):
            response_log = ResponseLog()
            response_log.add_response(response)
            self.log_access(request, response_log)

    def log_exception_access(
        self, request: Dict[str, Any], exception: BaseException
    ) -> None:
        response_log = ResponseLog()
        response_log.add_exception(exception)
        if not self.is_private_request(request):
            self.log_access(request, response_log)
        else:
            self.log_access(
                {request_id_field_name: request[request_id_field_name]}, response_log
            )
