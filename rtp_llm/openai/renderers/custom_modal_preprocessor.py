from typing import Any, List, Optional

import torch


class CustomModalPreprocessorInterface:
    @torch.inference_mode()
    def custom_modal_preprocess(self, batch_data: Any):
        raise NotImplementedError()
