import os
import sys
import json
import time
import logging
import logging.config
import traceback
from typing import Generator, Union, Any, Dict, List, AsyncGenerator, Optional
from pydantic import BaseModel

class VersionInfo(BaseModel):
    models_info: Optional[Dict[str, str]] = None
    peft_info: Optional[Dict[str, Any]] = None
    sampler_info: Optional[Dict[str, Any]] = None

