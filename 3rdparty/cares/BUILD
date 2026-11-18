# Vendored c-ares build support files.
#
# The @cares external repository (see deps/git.bzl) fetches a c-ares git
# snapshot that predates configure output, so the generated headers
# (ares_build.h and ares_config.h) cannot come from the archive. They are
# vendored here, paired with the snapshot commit
# 3be1924221e1326df520f8498d704a5c4c8d0cce, and copied into the external repo
# by genrules in cares.BUILD.
#
# Header provenance: grpc@109c570727c3089fef655edcdd0dd02cc5958010
# third_party/cares/ares_build.h and config_linux/ares_config.h (the same
# snapshot grpc builds against).

exports_files([
    "ares_build.h",
    "ares_config_linux.h",
    "cares.BUILD",
])
