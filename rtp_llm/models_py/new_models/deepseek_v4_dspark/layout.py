"""Checkpoint routing for the three-stage DeepSeek V4 DSpARK proposer."""

from rtp_llm.models_py.new_models.deepseek_v4.checkpoint_namespace import (
    V4CheckpointNamespace,
)


class DSparkV4CheckpointLayout:
    """Select only the draft role's live tensors from a joint target/draft index.

    The target and draft share embed/head. The loader subtracts bound loans
    from this filter before selecting shards; it still needs to see those two
    names if the proposal model is loaded independently.
    """

    _shared = frozenset(("embed.weight", "head.weight"))
    _projection = frozenset(
        ("mtp.0.main_proj.weight", "mtp.0.main_proj.scale", "mtp.0.main_norm.weight")
    )
    _sampler = frozenset(
        ("mtp.2.markov_head.markov_w1.weight", "mtp.2.markov_head.markov_w2.weight")
    )

    def __init__(self, commit_only):
        if type(commit_only) is not bool:
            raise TypeError("commit_only must be bool")
        self.commit_only = commit_only
        self._backbone = V4CheckpointNamespace(
            layer_prefix="mtp",
            global_prefix="mtp.2.",
            layer_count=3,
            commit_only=commit_only,
        )

    def auxiliary(self, name):
        return name in self._projection or (
            not self.commit_only and name in self._sampler
        )

    def includes(self, name):
        # The released draft contains a training confidence predictor in its
        # final stage. Inference samples with the Markov tables, not this head.
        # Leave unexpected stage tensors visible to strict dispatch.
        # Embedding and LM head are shared target globals, unlike the final
        # stage's own norm and HC head. Never load a second copy into them.
        if name in (
            "mtp.2.confidence_head.proj.weight",
            "mtp.2.embed.weight",
            "mtp.2.head.weight",
        ):
            return False
        if name in self._shared or self.auxiliary(name):
            return True
        return self._backbone.accepts(name)

    def mapped(self, name, *, auxiliary_mapper, backbone_mapper):
        if self.auxiliary(name):
            return "auxiliary." + auxiliary_mapper.map_name(name)
        return "v4." + backbone_mapper.map_name(name)
