from typing import Any, Dict, Optional

from pydantic import BaseModel


class VersionInfo(BaseModel):
    models_info: Optional[Dict[str, str]] = None
    peft_info: Optional[Dict[str, Any]] = None
    sampler_info: Optional[Dict[str, Any]] = None
