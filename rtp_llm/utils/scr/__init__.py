"""Snapshot/checkpoint/restore (SCR) support.

Epsilon owns the control plane: RTP-LLM never invokes a controller and never
performs dump/restore itself.  A process only registers its state and
announces a passive arrival at the barrier.

Import from this package instead of the individual modules; the file layout
is an implementation detail.

    template_utils   gate, manifest, Epsilon adapter, registration, arrival
    lifecycle        template lifecycle callbacks
    restore_context  the restored-Pod identity carried across CRIU
    runtime_fixup    re-derive runtime state after a restore
    endpoint         endpoint resolution from the restore manifest
    network          VIP topology and single-Pod loopback addressing
"""

from rtp_llm.utils.scr.lifecycle import (
    CallbackHook,
    TemplateLifecycle,
    get_template_lifecycle,
    template_phase_active,
)
from rtp_llm.utils.scr.template_utils import (
    ScrParticipantManifest,
    arrive_scr_checkpoint_barrier,
    arrive_scr_template_barrier,
    build_scr_participant_manifest,
    configure_scr_environment,
    epsilon_backend_mode,
    get_scr_barrier_return_time,
    is_scr_enabled,
    is_scr_template_phase_active,
    register_for_scr,
)

__all__ = [
    "CallbackHook",
    "ScrParticipantManifest",
    "TemplateLifecycle",
    "arrive_scr_checkpoint_barrier",
    "arrive_scr_template_barrier",
    "build_scr_participant_manifest",
    "configure_scr_environment",
    "epsilon_backend_mode",
    "get_scr_barrier_return_time",
    "get_template_lifecycle",
    "is_scr_enabled",
    "is_scr_template_phase_active",
    "register_for_scr",
    "template_phase_active",
]
