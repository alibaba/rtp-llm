"""Snapshot/checkpoint/restore (SCR) support.

Epsilon owns the control plane: RTP-LLM never invokes a controller and never
performs dump/restore itself.  A process only registers its state and
announces a passive arrival at the barrier.

Import the submodule you need (``rtp_llm.utils.scr.lifecycle``,
``rtp_llm.utils.scr.template_utils``, ...); this package intentionally does
not re-export them, so importing the lifecycle does not pull in the Epsilon
integration or its torch dependencies.
"""
