# The paired CI runner supplies the OldFST fixture locally before Bazel
# analyzes the NewFST targets. No captured tensors are present in Git.
package(default_visibility = ["//rtp_llm/test/smoke:__pkg__"])

filegroup(
    name = "v41draft",
    srcs = glob(["v41draft_p0.safetensors", "v41draft_p1.safetensors", "v41draft_d0.safetensors"], allow_empty = True),
)

filegroup(
    name = "v4",
    srcs = glob(["v4_r0.safetensors", "v4_r1.safetensors"], allow_empty = True),
)

filegroup(
    name = "v41",
    srcs = glob(["v41_r0.safetensors", "v41_r1.safetensors"], allow_empty = True),
)
