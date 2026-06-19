# xgrammar built from source (mlc-ai/xgrammar @ v0.2.2). Mirrors what
# xgrammar's CMakeLists does:
#   file(GLOB_RECURSE cpp/*.cc EXCLUDE cpp/nanobind/*)
#   add_library(xgrammar STATIC ${SOURCES})
#   target_include_directories include + cpp + 3rdparty/picojson +
#                              3rdparty/dlpack/include
#   compile_definitions XGRAMMAR_ENABLE_CPPTRACE=0   (disable optional tracing)
#
# Submodules picojson and dlpack are pulled by init_submodules=True in
# internal_source/deps/git.bzl. cpptrace + googletest submodules are also
# pulled but unused — the .a we build never references them.

# Public headers — what RTP-LLM C++ code includes via #include <xgrammar/...>
# and <dlpack/dlpack.h> (transitively via xgrammar/matcher.h).
cc_library(
    name = "xgrammar_headers",
    hdrs = glob([
        "include/xgrammar/*.h",
        "3rdparty/dlpack/include/dlpack/*.h",
    ]),
    includes = [
        "include",
        "3rdparty/dlpack/include",
    ],
    visibility = ["//visibility:public"],
)

# Internal headers under cpp/ that xgrammar source files #include directly
# (e.g. "compiled_grammar_impl.h", "grammar_impl.h"). Not exposed to
# downstream — callers should use xgrammar/*.h public API only.
cc_library(
    name = "xgrammar_internal_headers",
    hdrs = glob([
        "cpp/*.h",
        "cpp/support/*.h",
        "3rdparty/picojson/picojson.h",
    ]),
    includes = [
        "cpp",
        "3rdparty/picojson",
    ],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "xgrammar",
    srcs = glob(
        [
            "cpp/*.cc",
            "cpp/support/*.cc",
        ],
        exclude = [
            "cpp/nanobind/**",
            # NB: cpp/testing.cc MUST be included — besides debug-only
            # _DebugParseEBNF / _PrintGrammarFSMs, it also defines
            # xgrammar::PrintTokenByIds, which grammar_matcher.cc and
            # compiled_grammar.cc reference from their operator<< debug
            # overloads. Excluding it yields a link-time undefined symbol.
        ],
    ),
    defines = [
        "XGRAMMAR_ENABLE_CPPTRACE=0",
    ],
    copts = [
        "-std=c++17",
        "-fexceptions",
        "-Wno-unused-variable",
        "-Wno-sign-compare",
    ],
    deps = [
        ":xgrammar_headers",
        ":xgrammar_internal_headers",
    ],
    linkstatic = True,
    visibility = ["//visibility:public"],
)
