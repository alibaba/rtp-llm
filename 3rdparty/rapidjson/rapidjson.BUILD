# Tencent RapidJSON (header-only).
# 仅供 @havenask 内 aios/autil 等目标通过 @rapidjson 引用；非 rtp_llm 业务代码的直接依赖。
# 勿因「源码未引用」删除 http.bzl 中对应 http_archive。
licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "rapidjson",
    hdrs = glob(["include/rapidjson/**/*.h"]),
    includes = ["include"],
)
