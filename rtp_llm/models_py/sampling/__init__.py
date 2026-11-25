from dataclasses import dataclass


@dataclass(frozen=True)
class RejectionOptions:
    graph: bool = True
    fuse_draft_probs_log: bool = True


rejection_options = RejectionOptions()
