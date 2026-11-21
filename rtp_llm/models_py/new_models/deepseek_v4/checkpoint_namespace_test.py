import unittest

from rtp_llm.models_py.new_models.deepseek_v4.checkpoint_namespace import (
    V4CheckpointNamespace,
)


class V4CheckpointNamespaceTest(unittest.TestCase):
    def test_target_excludes_other_roles_but_preserves_unknown_local_leaf(self):
        owns = V4CheckpointNamespace(
            layer_prefix="layers.", global_prefix="", layer_count=61, commit_only=False
        ).accepts
        for name in (
            "embed.weight",
            "hc_head_base",
            "norm.weight",
            "layers.0.attn.attn_sink",
            "layers.60.ffn.experts.383.w3.scale",
            "layers.60.unrecognized_leaf",
        ):
            self.assertTrue(owns(name), name)
        for name in (
            "mtp.0.attn.wkv.weight",
            "layers.61.attn.wkv.weight",
            "layers.060.attn.wkv.weight",
            "layers.-1.attn.wkv.weight",
            "layers.60",
            "unused.weight",
        ):
            self.assertFalse(owns(name), name)

    def test_draft_proposal_and_commit_do_not_read_target(self):
        def predicate(commit):
            return V4CheckpointNamespace(
                layer_prefix="mtp",
                global_prefix="mtp.0.",
                layer_count=3,
                commit_only=commit,
            ).accepts

        proposal, commit = predicate(False), predicate(True)
        for name in (
            "mtp.0.attn.wkv.weight",
            "mtp.2.attn.wkv.scale",
            "mtp.1.attn.kv_norm.weight",
        ):
            self.assertTrue(commit(name), name)
            self.assertTrue(proposal(name), name)
        for name in ("mtp.0.head.weight", "mtp.2.ffn.experts.383.w2.weight"):
            self.assertFalse(commit(name), name)
            self.assertTrue(proposal(name), name)
        for name in (
            "head.weight",
            "layers.0.attn.wkv.weight",
            "mtp.3.attn.wkv.weight",
        ):
            self.assertFalse(commit(name), name)
            self.assertFalse(proposal(name), name)

    def test_invalid_layer_config(self):
        for prefix, count in (("", 1), (".", 1), ("layers", 0), ("layers", True)):
            with self.assertRaises(ValueError):
                V4CheckpointNamespace(
                    layer_prefix=prefix,
                    global_prefix="",
                    layer_count=count,
                    commit_only=False,
                )


if __name__ == "__main__":
    unittest.main()
