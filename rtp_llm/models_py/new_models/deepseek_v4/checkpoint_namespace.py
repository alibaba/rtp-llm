"""Checkpoint namespaces consumed by a DeepSeek V4 target or draft.

EP placement is handled separately by the loader. The model-wide file set
must be identical across ranks even when each rank consumes different experts.
"""


class V4CheckpointNamespace:
    """Select a backbone's keys without scanning all layer prefixes per key."""

    _GLOBAL_NAMES = (
        "embed.weight",
        "head.weight",
        "norm.weight",
        "hc_head_fn",
        "hc_head_base",
        "hc_head_scale",
    )
    _COMMIT_ONLY_NAMES = frozenset(
        ("attn.wkv.weight", "attn.wkv.scale", "attn.kv_norm.weight")
    )

    def __init__(
        self,
        *,
        layer_prefix: str,
        global_prefix: str,
        layer_count: int,
        commit_only: bool
    ):
        if not layer_prefix or not layer_prefix.rstrip("."):
            raise ValueError("DeepSeek V4 checkpoint layer prefix must not be empty")
        if type(layer_count) is not int or layer_count <= 0:
            raise ValueError("DeepSeek V4 checkpoint layer count must be positive")
        self._layer_prefix = layer_prefix.rstrip(".") + "."
        self._layer_ids = frozenset(map(str, range(layer_count)))
        self._commit_only = commit_only
        self._globals = (
            frozenset()
            if commit_only
            else frozenset(global_prefix + name for name in self._GLOBAL_NAMES)
        )

    def accepts(self, name: str) -> bool:
        if name in self._globals:
            return True
        if not name.startswith(self._layer_prefix):
            return False
        layer, dot, leaf = name[len(self._layer_prefix) :].partition(".")
        if not dot or layer not in self._layer_ids:
            return False
        return not self._commit_only or leaf in self._COMMIT_ONLY_NAMES
