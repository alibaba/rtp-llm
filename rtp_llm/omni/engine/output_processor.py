import base64
import io
import logging
import struct
from typing import Any, Dict, Optional

import torch

from rtp_llm.omni.engine.stage_connector import StageOutput

logger = logging.getLogger(__name__)


class OmniOutputProcessor:
    def assemble(
        self,
        stage_outputs: Dict[int, StageOutput],
        final_output_types: Dict[str, int],
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {}

        for output_type, stage_id in final_output_types.items():
            stage_out = stage_outputs.get(stage_id)
            if stage_out is None:
                continue

            if output_type == "text":
                result["text"] = stage_out.metadata.get("text", "")
            elif output_type == "audio":
                result["audio"] = {
                    "waveform": stage_out.audio_waveform,
                    "metadata": stage_out.metadata,
                }
            elif output_type == "image":
                result["image"] = {
                    "tensor": stage_out.image_tensor,
                    "metadata": stage_out.metadata,
                }

        return result

    @staticmethod
    def encode_audio_base64(
        waveform: torch.Tensor,
        sample_rate: int = 16000,
        format: str = "wav",
    ) -> str:
        audio_data = waveform.squeeze().cpu().numpy()
        buf = io.BytesIO()
        num_samples = len(audio_data)
        num_channels = 1
        sample_width = 4  # float32
        # WAV header
        buf.write(b"RIFF")
        data_size = num_samples * num_channels * sample_width
        buf.write(struct.pack("<I", 36 + data_size))
        buf.write(b"WAVE")
        buf.write(b"fmt ")
        buf.write(struct.pack("<I", 16))
        buf.write(struct.pack("<H", 3))  # IEEE float
        buf.write(struct.pack("<H", num_channels))
        buf.write(struct.pack("<I", sample_rate))
        buf.write(struct.pack("<I", sample_rate * num_channels * sample_width))
        buf.write(struct.pack("<H", num_channels * sample_width))
        buf.write(struct.pack("<H", sample_width * 8))
        buf.write(b"data")
        buf.write(struct.pack("<I", data_size))
        buf.write(audio_data.tobytes())
        return base64.b64encode(buf.getvalue()).decode("utf-8")
