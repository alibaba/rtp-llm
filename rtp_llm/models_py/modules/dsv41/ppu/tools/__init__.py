"""Offline driver scripts for the V4.1 PPU decode machine (L3, D/DP8).

``hw_probe.py`` is the standalone 8-rank hardware/collective probe; it is
dependency-free apart from torch and deliberately never imports rtp_llm.
``driver_decode.py`` (L3, next wave) will drive the decode orchestration
against a real fixture.
"""
