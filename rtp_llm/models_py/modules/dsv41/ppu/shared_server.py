"""PPU-local standard lazy registration before the normal server bootstrap."""

from rtp_llm.config.v41_runtime import v41_backend_for_config
from rtp_llm.model_factory_register import register_lazy_model
from rtp_llm.models_py.modules.dsv41.ppu.backend import model_path

if v41_backend_for_config() != "ppu" or model_path() != "shared":
    raise ValueError("shared DSpark bootstrap requires explicit PPU/shared")

register_lazy_model(
    "deepseek_v41_ppu_shared_dspark",
    "rtp_llm.models_py.modules.dsv41.ppu.shared_registration",
)

if __name__ == "__main__":
    from rtp_llm.start_server import main

    main()
