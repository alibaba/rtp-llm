load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")
load(":mirror.bzl", "rtp_github_archive_urls", "rtp_mirror_urls")

# Repo names created by this file; `scripts/rtpcli bazel mod-tidy` keeps root use_repo in sync.
GIT_DEPS_EXPORTS = [
    "KleidiAI",
    "boringssl",
    "com_github_cares_cares",
    "com_github_nanopb_nanopb",
    "com_google_absl",
    "com_google_googletest",
    "curl",
    "cutlass",
    "cutlass3.6",
    "cutlass4.0",
    "cutlass_h_moe",
    "flashinfer_cpp",
    "flashmla",
    "grpc",
    "havenask",
    "krb5-devel",
    "libcom_err-devel",
    "nacos_sdk_cpp",
    "rapidjson",
    "rules_cc",
    "six_archive",
    "xgrammar",
    "zlib_archive",
]

def git_deps():
    # All third-party archives go through our own bucket mirror (byte-for-byte codeload
    # tarballs, sha pinned); no git_repository remains inside despite the function name.
    http_archive(
        name = "rules_cc",
        urls = rtp_github_archive_urls("bazelbuild", "rules_cc", "1477dbab59b401daa94acedbeaefe79bf9112167"),
        sha256 = "b87996d308549fc3933f57a786004ef65b44b83fd63f1b0303a4bbc3fd26bbaf",
        strip_prefix = "rules_cc-1477dbab59b401daa94acedbeaefe79bf9112167",
        # Bazel 7.7.1 provides cc_toolchain_alias/cc_libc_top_alias but not the
        # cc_host_toolchain_alias added by this archive's BUILD file.
        patch_args = ["-p1"],
        patches = [
            "@rtp_llm//patches/rules_cc:0001-bazel7-remove-host-toolchain-alias.patch",
            "@rtp_llm//patches/rules_cc:0002-bazel7-unpack-toolchain-info.patch",
        ],
    )

    # rules_python is not declared here: under bzlmod it is provided by the bazel_dep +
    # archive_override in MODULE.bazel; a repo here would collide with the bazel_dep.

    http_archive(
        name = "cutlass",
        urls = rtp_github_archive_urls("NVIDIA", "cutlass", "80243e0b8c644f281e2beb0c20fe78cf7b267061"),
        sha256 = "89ddcee9b478b1f5ec7423883ed673f13fb6982a3e93a12d314a3cb5b244db70",
        strip_prefix = "cutlass-80243e0b8c644f281e2beb0c20fe78cf7b267061",
        build_file = "@rtp_llm//3rdparty/cutlass:cutlass.BUILD",
    )

    http_archive(
        name = "cutlass_h_moe",
        urls = rtp_github_archive_urls("NVIDIA", "cutlass", "19b4c5e065e7e5bbc8082dfc7dbd792bdac850fc"),
        sha256 = "ab83e74e64b80581470cdb801231f780a709555d9097d1e3c43744a8f461c358",
        strip_prefix = "cutlass-19b4c5e065e7e5bbc8082dfc7dbd792bdac850fc",
        build_file = "@rtp_llm//3rdparty/cutlass:cutlass.BUILD",
    )

    http_archive(
        name = "cutlass3.6",
        urls = rtp_github_archive_urls("NVIDIA", "cutlass", "cc3c29a81a140f7b97045718fb88eb0664c37bd7"),
        sha256 = "81af753601e8011bf9189710b8abfd931bc26832c2801a76601bbab79ed19e44",
        strip_prefix = "cutlass-cc3c29a81a140f7b97045718fb88eb0664c37bd7",
        build_file = "@rtp_llm//3rdparty/cutlass:cutlass.BUILD",
        patches = ["@rtp_llm//3rdparty/cutlass:0001-cuda12.4-compat.patch"],
    )

    http_archive(
        name = "cutlass4.0",
        urls = rtp_github_archive_urls("NVIDIA", "cutlass", "dc4817921edda44a549197ff3a9dcf5df0636e7b"),
        sha256 = "f2a3a9df5e6f010c8b02716aa2644a6f071827fafa606fac5f5241cab6a1ab56",
        strip_prefix = "cutlass-dc4817921edda44a549197ff3a9dcf5df0636e7b",
        build_file = "@rtp_llm//3rdparty/cutlass:cutlass.BUILD",
    )

    http_archive(
        name = "flashinfer_cpp",
        urls = rtp_github_archive_urls("flashinfer-ai", "flashinfer", "1c88d650eeec97be3a4dcebe4a9912d7785bc250"),
        sha256 = "9cf63637206224219396961c73964affd8cb6d18c56923695217b74efc7c8f6d",
        strip_prefix = "flashinfer-1c88d650eeec97be3a4dcebe4a9912d7785bc250",
        build_file = "@rtp_llm//3rdparty/flashinfer:flashinfer.BUILD",
        patches = [
            "@rtp_llm//3rdparty/flashinfer:0001-fix-compile.patch",
            "@rtp_llm//3rdparty/flashinfer:0002-dispatch-group-size.patch",
            "@rtp_llm//3rdparty/flashinfer:0003-tanh-compatibility.patch",
            "@rtp_llm//3rdparty/flashinfer:0005-update-add-mla-attn-test-impl-mla-write-kvcache.patch",
            "@rtp_llm//3rdparty/flashinfer:0006-add-mla-dispatch-inc.patch",
            "@rtp_llm//3rdparty/flashinfer:0007-fix-nan.patch",
            "@rtp_llm//3rdparty/flashinfer:0008-enable-pdl.patch",
            "@rtp_llm//3rdparty/flashinfer:0009-sp-sample.patch",
            "@rtp_llm//3rdparty/flashinfer:0010-silu-mul-vec-size.patch",
        ],
    )

    http_archive(
        name = "flashmla",
        urls = rtp_github_archive_urls("deepseek-ai", "FlashMLA", "b31bfe72a83ea205467b3271a5845440a03ed7cb"),
        sha256 = "6a316051df12d503198f9a709590bba69e14d5baef1f59c583a2d8bf9d50b300",
        strip_prefix = "FlashMLA-b31bfe72a83ea205467b3271a5845440a03ed7cb",
        build_file = "@rtp_llm//3rdparty/flashmla:flashmla.BUILD",
        patches = [
            "@rtp_llm//3rdparty/flashmla:0001-add-interface.patch",
        ],
    )

    # xgrammar's picojson/dlpack declarations are all in-tree, so the tarball (no
    # submodules) is missing nothing.
    http_archive(
        name = "xgrammar",
        urls = rtp_github_archive_urls("mlc-ai", "xgrammar", "60fc70ee4e0842eecc81fdd1941f778b1bd8107f"),
        sha256 = "fdd07b18c234138615752a68be0c0d1cfd6160a8f65df8ed7f18e28a0ae67ccb",
        strip_prefix = "xgrammar-60fc70ee4e0842eecc81fdd1941f778b1bd8107f",
        build_file = "@rtp_llm//3rdparty/xgrammar:xgrammar.BUILD",
    )

    http_archive(
        name = "com_google_googletest",
        urls = rtp_github_archive_urls("google", "googletest", "f8d7d77c06936315286eb55f8de22cd23c188571"),
        sha256 = "7ff5db23de232a39cbb5c9f5143c355885e30ac596161a6b9fc50c4538bfbf01",
        strip_prefix = "googletest-f8d7d77c06936315286eb55f8de22cd23c188571",
    )

    http_archive(
        name = "com_github_nanopb_nanopb",
        sha256 = "8bbbb1e78d4ddb0a1919276924ab10d11b631df48b657d960e0c795a25515735",
        build_file = "@grpc//third_party:nanopb.BUILD",
        strip_prefix = "nanopb-f8ac463766281625ad710900479130c7fcb4d63b",
        urls = rtp_github_archive_urls("nanopb", "nanopb", "f8ac463766281625ad710900479130c7fcb4d63b"),
    )

    http_archive(
        name = "six_archive",
        build_file = "@rtp_llm//3rdparty/six:six.BUILD",
        sha256 = "105f8d68616f8248e24bf0e9372ef04d3cc10104f1980f54d57b2ce73a5ad56a",
        strip_prefix = "six-1.10.0",
        urls = rtp_mirror_urls("archives/pypi/six/six-1.10.0.tar.gz"),
    )

    http_archive(
        name = "zlib_archive",
        build_file = "@rtp_llm//3rdparty/zlib:zlib.BUILD",
        strip_prefix = "zlib-1.2.11",
        urls = rtp_mirror_urls("archives/zlib.net/zlib-1.2.11.tar.gz"),
        sha256 = "c3e5e9fdd5004dcb542feda5ee4f0ff0744628baf8ed2dd5d66f8ca1197cb1a1",
    )

    http_archive(
        name = "com_google_absl",
        urls = rtp_github_archive_urls("abseil", "abseil-cpp", "6f9d96a1f41439ac172ee2ef7ccd8edf0e5d068c"),
        sha256 = "62c27e7a633e965a2f40ff16b487c3b778eae440bab64cad83b34ef1cbe3aa93",
        strip_prefix = "abseil-cpp-6f9d96a1f41439ac172ee2ef7ccd8edf0e5d068c",
        patch_cmds = [
            "sed -i -e 's/^#define ABSL_OPTION_USE_STD_STRING_VIEW 2/#define ABSL_OPTION_USE_STD_STRING_VIEW 0/' 'absl/base/options.h'",
            "sed 's$@bazel_tools//platforms:(linux|osx|windows|android|freebsd|ios|os)$@platforms//os:\\1$' -E -i absl/BUILD.bazel",
            "sed 's$@bazel_tools//platforms:(cpu|x86_32|x86_64|ppc|arm|aarch64|s390x)$@platforms//cpu:\\1$' -i -E absl/BUILD.bazel",
            "sed 's$@bazel_tools//platforms:(linux|osx|windows|android|freebsd|ios|os)$@platforms//os:\\1$' -E -i absl/time/internal/cctz/BUILD.bazel",
            "sed 's$@bazel_tools//platforms:(cpu|x86_32|x86_64|ppc|arm|aarch64|s390x)$@platforms//cpu:\\1$' -i -E absl/time/internal/cctz/BUILD.bazel",
        ],
    )

    # com_google_protobuf is not declared here: native.local_repository is illegal inside a
    # module_extension; it is a top-level local_repository in MODULE.bazel.

    http_archive(
        name = "rapidjson",
        urls = rtp_github_archive_urls("Tencent", "rapidjson", "f54b0e47a08782a6131cc3d60f94d038fa6e0a51"),
        sha256 = "4a76453d36770c9628d7d175a2e9baccbfbd2169ced44f0cb72e86c5f5f2f7cd",
        strip_prefix = "rapidjson-f54b0e47a08782a6131cc3d60f94d038fa6e0a51",
        patches = ["@rtp_llm//3rdparty/rapidjson:0001-document_h.patch"],
        build_file = "@rtp_llm//3rdparty/rapidjson:rapidjson.BUILD",
    )

    http_archive(
        name = "havenask",
        urls = rtp_github_archive_urls("alibaba", "havenask", "3c973500afbd40933eb0a80cfdfb6592274377fb"),
        sha256 = "e03d63fa06095b612c5ba77e6b668dba4102ee90fdc79f7b45df545e64893b8b",
        strip_prefix = "havenask-3c973500afbd40933eb0a80cfdfb6592274377fb",
        patches=[
            "@rtp_llm//patches/havenask:havenask.patch",
            "@rtp_llm//patches/havenask:anet.patch",
            "@rtp_llm//patches/havenask:0001-fix-PrometheusSink-need-header.patch"
        ],
        build_file = "@rtp_llm//3rdparty/kmonitor:kmonitor.BUILD",
    )

    http_archive(
        name = "nacos_sdk_cpp",
        urls = rtp_github_archive_urls("nacos-group", "nacos-sdk-cpp", "2b4104d2524776dff236a228ad2abff4676fb916"),
        sha256 = "7c020f763b9af9706e84da42250146eb84bfd359c7286f7c1e1aa9a5be42d72d",
        strip_prefix = "nacos-sdk-cpp-2b4104d2524776dff236a228ad2abff4676fb916",
        patches = [
            "@rtp_llm//patches/nacos_sdk_cpp:nacos-compile.patch",
        ],
        build_file = "@rtp_llm//3rdparty/nacos_sdk_cpp:nacos_sdk_cpp.BUILD"
    )

    http_archive(
        name = "curl",
        build_file = "@rtp_llm//3rdparty/curl:curl.BUILD",
        sha256 = "e9c37986337743f37fd14fe8737f246e97aec94b39d1b71e8a5973f72a9fc4f5",
        strip_prefix = "curl-7.60.0",
        urls = rtp_mirror_urls("archives/curl.haxx.se/curl-7.60.0.tar.gz"),
    )

    http_archive(
        name = "grpc",
        urls = rtp_github_archive_urls("grpc", "grpc", "109c570727c3089fef655edcdd0dd02cc5958010"),
        sha256 = "ddd5c9c42bc609108c2e9494e9cfa34ea42d0efd0eb4b183db8a4124dabdc1c2",
        strip_prefix = "grpc-109c570727c3089fef655edcdd0dd02cc5958010",
        patches = [
            "@rtp_llm//patches/grpc:0001-Rename-gettid-functions.patch",
            "@rtp_llm//patches/grpc:0002-retire-external-binds.patch",
        ],
    )

    http_archive(
        name = "com_github_cares_cares",
        build_file = "@grpc//third_party:cares/cares.BUILD",
        sha256 = "e69e33fd40a254fcf00d76efa76776d45f960e34307bd9cea9df93ef79a933f1",
        strip_prefix = "c-ares-3be1924221e1326df520f8498d704a5c4c8d0cce",
        urls = rtp_github_archive_urls(
            "c-ares",
            "c-ares",
            "3be1924221e1326df520f8498d704a5c4c8d0cce",
        ),
    )

    # KleidiAI: upstream git.gitlab.arm.com TLS is incompatible with the intranet, so it is
    # bucket-mirrored from the same commit of the official GitHub mirror ARM-software/kleidiai
    # (a git SHA is content-addressed, so the tree is byte-identical to the gitlab side).
    http_archive(
        name = "KleidiAI",
        urls = rtp_github_archive_urls("ARM-software", "kleidiai", "2d160cf675d6df7068a17da07ec6218fd9478541"),
        sha256 = "ec6c94265835d5b362f8c17cfd70ce1363042b9dbe83d8341544ad8870376d16",
        strip_prefix = "kleidiai-2d160cf675d6df7068a17da07ec6218fd9478541",
        patch_args = ["-p1"],
        patches = ["@rtp_llm//patches/kai:0001-add-a8w4-fp16-support.patch"],
    )

    http_archive(
        name = "boringssl",
        sha256 = "1188e29000013ed6517168600fc35a010d58c5d321846d6a6dfee74e4c788b45",
        strip_prefix = "boringssl-7f634429a04abc48e2eb041c81c5235816c96514",
        urls = rtp_github_archive_urls("google", "boringssl", "7f634429a04abc48e2eb041c81c5235816c96514"),
    )

    # CentOS 7 is EOL; mirrors may be taken down or change bytes at any time, so both rpms
    # are bucket-mirrored (sha256 untouched).
    http_file(
        name = "krb5-devel",
        urls = rtp_mirror_urls("archives/mirrors.aliyun.com/centos/7/os/x86_64/Packages/krb5-devel-1.15.1-50.el7.x86_64.rpm"),
        sha256 = "75069ac38fed957b70ea1de5e2824e6a77468e9745a3a828d47a02bab727ba11",
    )

    http_file(
        name = "libcom_err-devel",
        urls = rtp_mirror_urls("archives/mirrors.aliyun.com/centos/7/os/x86_64/Packages/libcom_err-devel-1.42.9-19.el7.x86_64.rpm"),
        sha256 = "3a14db2d86490211494bb142139121da838160fb7ba28d46cd01568b0173969c",
    )
