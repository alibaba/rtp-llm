from enum import Enum, IntEnum


class ExceptionCategory(Enum):
    BAD_REQUEST = "bad_request"
    TOO_LONG = "too_long"
    UNSUPPORTED = "unsupported"
    CAPACITY = "capacity"
    TIMEOUT = "timeout"
    INVALID_OUTPUT = "invalid_output"
    CANCELLED = "cancelled"
    INTERNAL = "internal"


class ExceptionType(IntEnum):
    def __new__(
        cls,
        value: int,
        category: ExceptionCategory = ExceptionCategory.INTERNAL,
    ):
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj._category = category
        return obj

    # Old error codes, remain compatible
    TRAFFIC_LIMIT_ERROR = 429, ExceptionCategory.CAPACITY
    CONCURRENCY_LIMIT_ERROR = 409, ExceptionCategory.CAPACITY
    CANCELLED_ERROR = 499, ExceptionCategory.CANCELLED
    ERROR_INPUT_FORMAT_ERROR = 507, ExceptionCategory.BAD_REQUEST
    NO_PROMPT_ERROR = 509, ExceptionCategory.BAD_REQUEST
    EMPTY_PROMPT_ERROR = 510, ExceptionCategory.BAD_REQUEST
    LONG_PROMPT_ERROR = 511, ExceptionCategory.TOO_LONG
    UNKNOWN_ERROR = 514
    UNSUPPORTED_OPERATION = 515, ExceptionCategory.UNSUPPORTED
    UPDATE_ERROR = 601
    MALLOC_ERROR = 602
    GENERATE_TIMEOUT = 603, ExceptionCategory.TIMEOUT
    ERROR_GENERATE_CONFIG_FORMAT = 604, ExceptionCategory.BAD_REQUEST
    INVALID_PARAMS = 605, ExceptionCategory.BAD_REQUEST
    EXECUTION_EXCEPTION = 606
    EXCEEDS_KV_CACHE_MAX_LEN = 607, ExceptionCategory.TOO_LONG

    # Error codes starting from 8000 can be retried
    CANCELLED = 8100, ExceptionCategory.CANCELLED
    OUT_OF_VOCAB_RANGE = 8101, ExceptionCategory.INVALID_OUTPUT
    OUTPUT_QUEUE_FULL = 8102, ExceptionCategory.CAPACITY
    OUTPUT_QUEUE_IS_EMPTY = 8103
    FINISHED = 8104

    # rpc error
    GET_HOST_FAILED = 8200
    GET_CONNECTION_FAILED = 8201
    CONNECT_FAILED = 8202
    CONNECT_TIMEOUT = 8203, ExceptionCategory.TIMEOUT
    DEADLINE_EXCEEDED = 8204, ExceptionCategory.TIMEOUT
    CONNECTION_RESET_BY_PEER = 8205
    REMOTE_ALLOCATE_RESOURCE_WRITE_FAILED = 8206
    REMOTE_ALLOCATE_RESOURCE_READ_FAILED = 8207
    REMOTE_LOAD_KV_CACHE_FAILED = 8208
    REMOTE_GENERATE_FAILED = 8209
    RPC_FINISH_FAILED = 8210
    DECODE_MALLOC_FAILED = 8211
    LOAD_KV_CACHE_FAILED = 8212
    WAIT_TO_RUN_TIMEOUT = 8213, ExceptionCategory.TIMEOUT
    KEEP_ALIVE_TIMEOUT = 8214, ExceptionCategory.TIMEOUT

    # load cache error
    LOAD_CACHE_TIMEOUT = 8300, ExceptionCategory.TIMEOUT
    CACHE_STORE_PUSH_ITEM_FAILED = 8301
    CACHE_STORE_LOAD_CONNECT_FAILED = 8302
    CACHE_STORE_LOAD_SEND_REQUEST_FAILED = 8303
    CACHE_STORE_CALL_PREFILL_TIMEOUT = 8304, ExceptionCategory.TIMEOUT
    CACHE_STORE_LOAD_RDMA_CONNECT_FAILED = 8305
    CACHE_STORE_LOAD_RDMA_WRITE_FAILED = 8306
    CACHE_STORE_LOAD_BUFFER_TIMEOUT = 8307, ExceptionCategory.TIMEOUT
    CACHE_STORE_LOAD_UNKNOWN_ERROR = 8308
    CACHE_STORE_STORE_FAILED = 8309

    # p2p connector error
    P2P_CONNECTOR_CALL_PREFILL_FAILED = 8310
    P2P_CONNECTOR_LOAD_FROM_PREFILL_FAILED = 8311
    P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED = 8312
    P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED = 8313
    P2P_CONNECTOR_SCHEDULER_FILL_RESPONSE_FAILED = 8314
    P2P_CONNECTOR_WORKER_ASYMMETRIC_TP_FAILED = 8315
    P2P_CONNECTOR_WORKER_HANDLE_READ_TIMEOUT = 8316, ExceptionCategory.TIMEOUT
    P2P_CONNECTOR_WORKER_HANDLE_READ_CANCELLED = 8317
    P2P_CONNECTOR_WORKER_HANDLE_READ_TRANSFER_FAILED = 8318
    P2P_CONNECTOR_WORKER_READ_TRANSFER_RDMA_FAILED = 8319
    P2P_CONNECTOR_WORKER_READ_BUFFER_MISMATCH = 8320
    P2P_CONNECTOR_WORKER_HANDLE_READ_TRANSFER_TIMEOUT = 8321, (
        ExceptionCategory.TIMEOUT
    )
    P2P_CONNECTOR_WORKER_READ_FAILED = 8322
    P2P_CONNECTOR_WORKER_READ_CANCELLED = 8323
    P2P_CONNECTOR_WORKER_READ_TIMEOUT = 8324, ExceptionCategory.TIMEOUT
    P2P_CONNECTOR_WORKER_READ_TRANSFER_NOT_DONE = 8325

    # master error
    MASTER_NO_AVAILABLE_WORKER = 8400, ExceptionCategory.CAPACITY
    MASTER_NO_PREFILL_WORKER = 8402, ExceptionCategory.CAPACITY
    MASTER_NO_DECODE_WORKER = 8403, ExceptionCategory.CAPACITY
    MASTER_NO_PDFUSION_WORKER = 8404, ExceptionCategory.CAPACITY
    MASTER_NO_VIT_WORKER = 8405, ExceptionCategory.CAPACITY
    MASTER_INVALID_REQUEST = 8406, ExceptionCategory.BAD_REQUEST

    # route error
    ROUTE_ERROR = 8500, ExceptionCategory.CAPACITY
    ROUTER_QUEUE_FULL = 8502, ExceptionCategory.CAPACITY
    ROUTER_QUEUE_TIMEOUT = 8503, ExceptionCategory.TIMEOUT
    ROUTER_REQUEST_CANCELLED = 8504, ExceptionCategory.CANCELLED

    # multimodal error
    MM_LONG_PROMPT_ERROR = 901, ExceptionCategory.TOO_LONG
    MM_WRONG_FORMAT_ERROR = 902, ExceptionCategory.BAD_REQUEST
    MM_PROCESS_ERROR = 903
    MM_EMPTY_ENGINE_ERROR = 904
    MM_NOT_SUPPORTED_ERROR = 905, ExceptionCategory.UNSUPPORTED
    MM_DOWNLOAD_FAILED = 906

    @classmethod
    def from_value(cls, value):
        """根据给定的值返回对应的枚举名称，或者引发值错误。"""
        try:
            return cls(value).name  # 获取对应的枚举名
        except ValueError:
            raise ValueError(f"{value} is not a valid ExceptionType")

    @property
    def category(self) -> ExceptionCategory:
        return self._category


class FtRuntimeException(Exception):
    def __init__(self, exception_type: ExceptionType, message: str):
        self.exception_type = exception_type
        self.message = message
        super().__init__(self.message)
