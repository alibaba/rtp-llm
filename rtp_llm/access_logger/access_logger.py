import logging
import time
from typing import Any, Dict, List, Optional

from rtp_llm.access_logger.json_util import dump_json
from rtp_llm.access_logger.log_utils import get_handler
from rtp_llm.access_logger.py_access_log import PyAccessLog, RequestLog, ResponseLog
from rtp_llm.structure.request_extractor import request_id_field_name
from rtp_llm.utils.base_model_datatypes import MultimodalInput

ACCESS_LOGGER_NAME = "access_logger"
QUERY_ACCESS_LOGGER_NAME = "query_access_logger"
MM_ACCESS_LOGGER_NAME = "mm_access_logger"
MM_QUERY_ACCESS_LOGGER_NAME = "mm_query_access_logger"

def init_logger(logger_name: str, filename: str,log_path: str, backup_count: int, rank_id: Optional[int] = None, server_id: Optional[int] = None, async_mode: bool = True) -> None:
    access_logger = logging.getLogger(logger_name)
    handler = get_handler(filename, log_path, backup_count, rank_id, server_id, async_mode)
    formatter = logging.Formatter("%(message)s")
    access_logger.handlers.clear()
    access_logger.parent = None
    if handler is not None:
        handler.setFormatter(formatter)
        access_logger.addHandler(handler)


class AccessLogger:
    def __init__(self, log_path: str, backup_count: int, rank_id: Optional[int] = None, server_id: Optional[int] = None, async_mode: bool = True) -> None:
        init_logger(ACCESS_LOGGER_NAME, "access.log", log_path, backup_count, rank_id, server_id, async_mode)
        init_logger(QUERY_ACCESS_LOGGER_NAME, "query_access.log", log_path, backup_count, rank_id, server_id, async_mode)
        self.logger = logging.getLogger(ACCESS_LOGGER_NAME)
        self.query_logger = logging.getLogger(QUERY_ACCESS_LOGGER_NAME)
        self.async_mode = async_mode
        self.rank_id = rank_id
        self.server_id = server_id
        logging.info(f"AccessLogger created: async_mode={async_mode}, rank_id={rank_id}, server_id={server_id}")

    @staticmethod
    def is_private_request(request: Dict[str, Any]):
        return request.get("private_request", False)

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


class MMAccessLogger(AccessLogger):
    def __init__(self, log_path: str, backup_count: int, rank_id: Optional[int] = None, server_id: Optional[int] = None, async_mode: bool = True) -> None:
        init_logger(MM_ACCESS_LOGGER_NAME, "mm_access.log", log_path, backup_count, rank_id, server_id, async_mode)
        init_logger(MM_QUERY_ACCESS_LOGGER_NAME, "mm_query_access.log", log_path, backup_count, rank_id, server_id, async_mode)
        self.logger = logging.getLogger(MM_ACCESS_LOGGER_NAME)
        self.query_logger = logging.getLogger(MM_QUERY_ACCESS_LOGGER_NAME)

    def log(
        self,
        logger,
        request: List[MultimodalInput],
        exception: Optional[BaseException] = None,
        response: Optional[Any] = None,
    ) -> None:
        current_time = time.time()
        local_time = time.localtime(current_time)
        log_time = (
            time.strftime("%Y-%m-%d %H:%M:%S", local_time)
            + f".{int((current_time % 1) * 1000):03d}"
        )
        logger.info(
            dump_json(
                {
                    "query": request,
                    "log_time": log_time,
                    "exception": exception,
                    "response": response,
                }
            )
        )

    def log_query_access(self, mm_inputs: List[MultimodalInput]) -> None:
        self.log(self.query_logger, mm_inputs)

    def log_exception_access(
        self, mm_inputs: List[MultimodalInput], exception: BaseException
    ) -> None:
        self.log(self.logger, mm_inputs, exception=exception)

    def log_success_access(
        self, mm_inputs: List[MultimodalInput], response: Any
    ) -> None:
        self.log(self.logger, mm_inputs, response=response)
