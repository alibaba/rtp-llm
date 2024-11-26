from enum import IntEnum

class ExceptionType(IntEnum):
    CONCURRENCY_LIMIT_ERROR = 409
    CANCELLED_ERROR = 499
    ERROR_INPUT_FORMAT_ERROR = 507
    NO_PROMPT_ERROR = 509
    EMPTY_PROMPT_ERROR = 510
    LONG_PROMPT_ERROR = 511
    ERROR_STOP_LIST_FORMAT = 512
    EXCEEDS_KV_CACHE_MAX_LEN = 513
    UNKNOWN_ERROR = 514
    UNSUPPORTED_OPERATION = 515
    ERROR_GENERATE_CONFIG_FORMAT = 516
    UPDATE_ERROR = 601
    MALLOC_ERROR = 602
    GENERATE_TIMEOUT = 603
    CANCELLED = 604
    OUT_OF_VOCAB_RANGE = 605
    OUTPUT_QUEUE_FULL = 606
    OUTPUT_QUEUE_IS_EMPTY = 607

    # rpc error
    GET_HOST_FAILED = 704
    GET_CONNECTION_FAILED = 705
    CONNECT_FAILED = 706
    CONNECTION_RESET_BY_PEER = 707
    REMOTE_ALLOCATE_RESOURCE_FAILED = 708
    REMOTE_LOAD_KV_CACHE_FAILED = 709
    REMOTE_GENERATE_FAILED = 710
    RPC_FINISH_FAILED = 711
    DECODE_MALLOC_FAILED = 712
    LOAD_KV_CACHE_FAILED = 713

    # load cache error
    LOAD_CACHE_TIMEOUT = 810
    CACHE_STORE_LOAD_CONNECT_FAILED = 811
    CACHE_STORE_LOAD_SEND_REQUEST_FAILED = 812
    CACHE_STORE_CALL_PREFILL_TIMEOUT = 813
    CACHE_STORE_LOAD_RDMA_CONNECT_FAILED = 814
    CACHE_STORE_LOAD_RDMA_WRITE_FAILED = 815
    CACHE_STORE_LOAD_BUFFER_TIMEOUT = 816

    @classmethod
    def from_value(cls, value):
        """根据给定的值返回对应的枚举名称，或者引发值错误。"""
        try:
            return cls(value).name  # 获取对应的枚举名
        except ValueError:
            raise ValueError(f"{value} is not a valid ExceptionType")

class FtRuntimeException(Exception):
    def __init__(self, exception_type: ExceptionType, message: str):
        self.exception_type = exception_type
        self.message = message
        super().__init__(self.message)
