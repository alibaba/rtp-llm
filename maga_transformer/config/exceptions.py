from enum import IntEnum

class ExceptionType(IntEnum):
    ERROR_INPUT_FORMAT_ERROR = 507
    GPT_NOT_FOUND_ERROR = 508
    NO_PROMPT_ERROR = 509
    EMPTY_PROMPT_ERROR = 510    
    LONG_PROMPT_ERROR = 511       
    ERROR_STOP_LIST_FORMAT = 512
    CONCURRENCY_LIMIT_ERROR = 513
    UNKNOWN_ERROR = 514
    UNSUPPORTED_OPERATION = 515
    ERROR_GENERATE_CONFIG_FORMAT = 516
    

class FtRuntimeException(Exception):
    def __init__(self, expcetion_type: ExceptionType, message: str):
        self.expcetion_type = expcetion_type
        self.message = message
        super().__init__(self.message)
