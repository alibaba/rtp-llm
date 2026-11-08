"""N7 startup-selected modes on the retained native role lifecycle."""

import os

from rtp_llm.models_py.modules.dsv41.ppu.test.native_shared_dspark_role import main
from rtp_llm.models_py.modules.dsv41.ppu.tools.driver_shared_dspark_modes import selected_mode


if __name__ == "__main__":
    main(mode=selected_mode(os.environ))
