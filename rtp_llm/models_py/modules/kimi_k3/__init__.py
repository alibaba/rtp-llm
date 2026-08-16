"""Kimi K3 modeling components.

``model_desc.kimi_k3`` owns the RTP model and decoder composition. KDA, MLA
and K3-specific mathematical primitives live in this package. Dense MLP and
sequence-parallel execution reuse framework modules under ``modules.hybrid``
and ``models_py.distributed``; KDA uses ``LinearCacheConverter`` directly.
"""
