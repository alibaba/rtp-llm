import logging
import os
import uuid


def initialize_pdfusion_coordination(py_env_configs) -> None:
    """Allocate one control identity per backend launch, before spawning ranks."""
    config = py_env_configs.runtime_config.fifo_scheduler_config
    adaptive = os.environ.get("PDFUSION_CADENCE_ADAPTIVE", "0")
    if (
        not adaptive.isascii()
        or not adaptive.isdecimal()
        or int(adaptive) not in (0, 1)
    ):
        raise ValueError("PDFUSION_CADENCE_ADAPTIVE must be 0 or 1")
    if config.pdfusion_coord_mode != "cadence" and int(adaptive) == 0:
        return

    # Even if the same config is reused, a new launch must not join an old group.
    # Pass this field through the existing config pickle, never per-rank env/PIDs.
    config.pdfusion_coord_group_id = uuid.uuid4().hex
    logging.info("PDFUSION_COORD_BOOTSTRAP group=%s", config.pdfusion_coord_group_id)
