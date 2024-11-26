import re
import os
import json
import logging
from enum import Enum
from typing import Optional, Union, Dict, Any, List, Set

def is_positive_integer(value):
    return isinstance(value, int) and value >= 0

def is_list_positive_integer(value):
    return (isinstance(value, list) and all([is_positive_integer(i) for i in value]))

def is_list_positive_integer_list(value):
    return (isinstance(value, list) and all([is_list_positive_integer(i) for i in value]))

def is_positive_float(value):
    return isinstance(value, float) and value >= 0

def is_list_positive_float(value):
    return isinstance(value, list) and all([is_positive_float(i) for i in value])

def is_positive_number(value):
    return isinstance(value, (float, int)) and value >= 0

def is_list_positive_number(value):
    return isinstance(value, list) and all([is_positive_number(i) for i in value])

def is_union_positive_integer(value):
    return is_positive_integer(value) or is_list_positive_integer(value)

def is_union_positive_number(value):
    return is_positive_number(value) or is_list_positive_number(value)

def check_optional(pred_func, value):
    return pred_func(value) if value else True

