"""Public PPU MHA fallbacks, loaded only when the attention slot is consumed."""


def register_attention(*, prefill_mha_imps, decode_mha_imps, **kwargs):
    from rtp_llm.device.device_type import DeviceType, get_device_type

    if get_device_type() != DeviceType.Ppu:
        return
    from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_cp_flashinfer import (
        CPFlashInferImpl,
    )
    from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
        PyFlashinferDecodeImpl,
        PyFlashinferHybridPrefillImpl,
        PyFlashinferPagedPrefillImpl,
        PyFlashinferPrefillImpl,
    )

    prefill_mha_imps.extend(
        [
            PyFlashinferPrefillImpl,
            PyFlashinferHybridPrefillImpl,
            PyFlashinferPagedPrefillImpl,
            CPFlashInferImpl,
        ]
    )
    decode_mha_imps.append(PyFlashinferDecodeImpl)
