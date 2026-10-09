"""Read C2 gang membership from the existing Pod annotations projection."""

import json
from pathlib import Path
from typing import Any


def read_c2_gang_info(annotation_path: str) -> dict[str, Any]:
    # Preserve the annotation format used by get_master_from_c2. Reopen on each
    # call so retries and restore observe the kubelet's updated projection.
    content = Path(annotation_path).read_text()
    infos = [x for x in content.split("\n") if "app.c2.io/biz-detail-ganginfo" in x]
    if len(infos) != 1:
        raise ValueError(f"ganginfo length is not equal to 1, actual: {infos}")
    gang_info = infos[0].replace("\\", "")
    return json.loads(gang_info[gang_info.index("=") + 2 : -1])
