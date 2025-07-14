import json
import logging
import logging.config
import os
import sys
import time
import traceback
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Union

from pydantic import BaseModel


class VersionInfo(BaseModel):
    models_info: Optional[Dict[str, str]] = None
    peft_info: Optional[Dict[str, Any]] = None
    sampler_info: Optional[Dict[str, Any]] = None
