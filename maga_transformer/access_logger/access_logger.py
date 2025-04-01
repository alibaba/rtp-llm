import os
import logging
from typing import Any, Union, Dict

from maga_transformer.access_logger.json_util import dump_json
from maga_transformer.access_logger.log_utils import get_handler
from maga_transformer.access_logger.py_access_log import RequestLog, ResponseLog, PyAccessLog
from maga_transformer.structure.request_extractor import request_id_field_name

ACCESS_LOGGER_NAME = 'access_logger'
QUERY_ACCESS_LOGGER_NAME = 'query_access_logger'

LOG_RESPONSE = int(os.environ.get('PY_INFERENCE_LOG_RESPONSE', '0')) == 1

def init_access_logger() -> None:
    access_logger = logging.getLogger(ACCESS_LOGGER_NAME)
    handler = get_handler('access.log')
    formatter = logging.Formatter('%(message)s')
    access_logger.handlers.clear()
    access_logger.parent = None
    if handler != None:
        handler.setFormatter(formatter)
        access_logger.addHandler(handler)

def init_query_access_logger() -> None:
    access_logger = logging.getLogger(QUERY_ACCESS_LOGGER_NAME)
    handler = get_handler('query_access.log')
    formatter = logging.Formatter('%(message)s')
    access_logger.handlers.clear()
    access_logger.parent = None
    if handler != None:
        handler.setFormatter(formatter)
        access_logger.addHandler(handler)

class AccessLogger():
    def __init__(self) -> None:
        init_access_logger()
        init_query_access_logger()
        self.logger = logging.getLogger(ACCESS_LOGGER_NAME)
        self.query_logger = logging.getLogger(QUERY_ACCESS_LOGGER_NAME)

    @staticmethod
    def is_private_request(request: Dict[str, Any]):
        return request.get('private_request', False)

    def log_access(self, request: Dict[str, Any], response: ResponseLog) -> None:
        request_log = RequestLog.from_request(request)
        access_log = PyAccessLog(request = request_log, response = response, id = request[request_id_field_name])
        self.logger.info(dump_json(access_log))

    def log_query_access(self, request: Dict[str, Any]) -> None:
        if not self.is_private_request(request):
            request_log = RequestLog.from_request(request)
            response_log = ResponseLog()
            access_log = PyAccessLog(request = request_log, response = response_log, id = request[request_id_field_name])
            self.query_logger.info(dump_json(access_log))

    def log_success_access(self, request: Dict[str, Any], response: Any) -> None:
        if not self.is_private_request(request):
            response_log = ResponseLog()
            if LOG_RESPONSE:
                response_log.add_response(response)
            self.log_access(request, response_log)

    def log_exception_access(self, request: Dict[str, Any], exception: BaseException) -> None:
        response_log = ResponseLog()
        response_log.add_exception(exception)
        if not self.is_private_request(request):
            self.log_access(request, response_log)
        else:
            self.log_access({request_id_field_name : request[request_id_field_name]}, response_log)

