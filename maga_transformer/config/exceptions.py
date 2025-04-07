from enum import IntEnum

class ExceptionType(IntEnum):
    # Old error codes, remain compatible
    CONCURRENCY_LIMIT_ERROR = 409
    CANCELLED_ERROR = 499
    ERROR_INPUT_FORMAT_ERROR = 507
    NO_PROMPT_ERROR = 509
    EMPTY_PROMPT_ERROR = 510
    LONG_PROMPT_ERROR = 511
    UNKNOWN_ERROR = 514
    UNSUPPORTED_OPERATION = 515
    UPDATE_ERROR = 601
    MALLOC_ERROR = 602
    GENERATE_TIMEOUT = 603
    ERROR_GENERATE_CONFIG_FORMAT = 604
    INVALID_PARAMS = 605
    EXECUTION_EXCEPTION = 606

    # Error codes starting from 8000 can be retried
    CANCELLED = 8100
    OUT_OF_VOCAB_RANGE = 8101
    OUTPUT_QUEUE_FULL = 8102
    OUTPUT_QUEUE_IS_EMPTY = 8103
    FINISHED = 8104
    EXCEEDS_KV_CACHE_MAX_LEN = 8105

    # rpc error
    GET_HOST_FAILED = 8200
    GET_CONNECTION_FAILED = 8201
    CONNECT_FAILED = 8202
    CONNECT_TIMEOUT = 8203
    DEADLINE_EXCEEDED = 8204
    CONNECTION_RESET_BY_PEER = 8205
    REMOTE_ALLOCATE_RESOURCE_WRITE_FAILED = 8206
    REMOTE_ALLOCATE_RESOURCE_READ_FAILED = 8207
    REMOTE_LOAD_KV_CACHE_FAILED = 8208
    REMOTE_GENERATE_FAILED = 8209
    RPC_FINISH_FAILED = 8210
    DECODE_MALLOC_FAILED = 8211
    LOAD_KV_CACHE_FAILED = 8212
    WAIT_TO_RUN_TIMEOUT = 8213

    # load cache error
    LOAD_CACHE_TIMEOUT = 8300
    CACHE_STORE_PUSH_ITEM_FAILED = 8301
    CACHE_STORE_LOAD_CONNECT_FAILED = 8302
    CACHE_STORE_LOAD_SEND_REQUEST_FAILED = 8303
    CACHE_STORE_CALL_PREFILL_TIMEOUT = 8304
    CACHE_STORE_LOAD_RDMA_CONNECT_FAILED = 8305
    CACHE_STORE_LOAD_RDMA_WRITE_FAILED = 8306
    CACHE_STORE_LOAD_BUFFER_TIMEOUT = 8307
    CACHE_STORE_LOAD_UNKNOWN_ERROR = 8308
    CACHE_STORE_STORE_FAILED = 8309

    # multimodal error
    MM_LONG_PROMPT_ERROR = 901
    MM_WRONG_FORMAT_ERROR = 902
    MM_PROCESS_ERROR = 903
    MM_EMPTY_ENGINE_ERROR = 904
    MM_NOT_SUPPORTED_ERROR = 905
    MM_DOWNLOAD_FAILED = 906

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
