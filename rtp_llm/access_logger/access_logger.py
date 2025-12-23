import logging
from typing import Any, Dict, Optional

from rtp_llm.access_logger.json_util import dump_json
from rtp_llm.access_logger.log_utils import get_handler
from rtp_llm.access_logger.py_access_log import PyAccessLog, RequestLog, ResponseLog
from rtp_llm.structure.request_extractor import request_id_field_name

ACCESS_LOGGER_NAME = "access_logger"
QUERY_ACCESS_LOGGER_NAME = "query_access_logger"


def init_access_logger(
    log_path: str,
    backup_count: int,
    rank_id: Optional[int] = None,
    server_id: Optional[int] = None,
    async_mode: bool = True,
) -> None:
    access_logger = logging.getLogger(ACCESS_LOGGER_NAME)
    handler = get_handler(
        "access.log", log_path, backup_count, rank_id, server_id, async_mode
    )
    formatter = logging.Formatter("%(message)s")
    access_logger.handlers.clear()
    access_logger.parent = None
    if handler is not None:
        handler.setFormatter(formatter)
        access_logger.addHandler(handler)


def init_query_access_logger(
    log_path: str,
    backup_count: int,
    rank_id: Optional[int] = None,
    server_id: Optional[int] = None,
    async_mode: bool = True,
) -> None:
    access_logger = logging.getLogger(QUERY_ACCESS_LOGGER_NAME)
    handler = get_handler(
        "query_access.log", log_path, backup_count, rank_id, server_id, async_mode
    )
    formatter = logging.Formatter("%(message)s")
    access_logger.handlers.clear()
    access_logger.parent = None
    if handler is not None:
        handler.setFormatter(formatter)
        access_logger.addHandler(handler)


class AccessLogger:
    def __init__(
        self,
        log_path: str,
        backup_count: int,
        rank_id: Optional[int] = None,
        server_id: Optional[int] = None,
        async_mode: bool = True,
    ) -> None:
        init_access_logger(log_path, backup_count, rank_id, server_id, async_mode)
        init_query_access_logger(log_path, backup_count, rank_id, server_id, async_mode)
        self.logger = logging.getLogger(ACCESS_LOGGER_NAME)
        self.query_logger = logging.getLogger(QUERY_ACCESS_LOGGER_NAME)
        self.async_mode = async_mode
        self.rank_id = rank_id
        self.server_id = server_id
        logging.info(
            f"AccessLogger created: async_mode={async_mode}, rank_id={rank_id}, server_id={server_id}"
        )

    @staticmethod
    def is_private_request(request: Dict[str, Any]):
        return request.get("private_request", False)

    def log_access(self, request: Dict[str, Any], response: ResponseLog) -> None:
        request_log = RequestLog.from_request(request)
        access_log = PyAccessLog(
            request=request_log,
            response=response,
            id=request.get(request_id_field_name, -1),
        )
        self.logger.info(dump_json(access_log))

    def log_query_access(self, request: Dict[str, Any]) -> None:
        if not self.is_private_request(request):
            request_log = RequestLog.from_request(request)
            response_log = ResponseLog()
            access_log = PyAccessLog(
                request=request_log,
                response=response_log,
                id=request.get(request_id_field_name, -1),
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
                {request_id_field_name: request.get(request_id_field_name, -1)},
                response_log,
            )
