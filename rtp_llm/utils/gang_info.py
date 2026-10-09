"""Read C2 gang membership from the existing Pod annotations projection."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rtp_llm.config.py_config_modules import DistributeConfig


class GangInfoReader:
    def __init__(self, annotation_path: str):
        self._annotation_path = Path(annotation_path)

    @classmethod
    def from_config(cls, config: "DistributeConfig") -> "GangInfoReader":
        return cls(config.gang_annocation_path)

    def read(self) -> dict[str, Any]:
        # Keep only the path, not an open file or cached rows: kubelet can replace
        # the projection while startup retries or a checkpointed process resumes.
        content = self._annotation_path.read_text()
        infos = [x for x in content.split("\n") if "app.c2.io/biz-detail-ganginfo" in x]
        if len(infos) != 1:
            raise ValueError(f"ganginfo length is not equal to 1, actual: {infos}")
        gang_info = infos[0].replace("\\", "")
        return json.loads(gang_info[gang_info.index("=") + 2 : -1])
