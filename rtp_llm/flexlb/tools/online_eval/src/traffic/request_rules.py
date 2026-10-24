"""Pure numeric rules shared by request content and execution boundaries."""
from input_contract import NumberRule

PRIORITY = NumberRule(integer=True, minimum=1, maximum=100)
JAVA_LENGTH = NumberRule(integer=True, minimum=1, maximum=(1 << 31) - 1)
