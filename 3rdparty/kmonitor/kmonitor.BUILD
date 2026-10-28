# 作为 git.bzl 中 @havenask new_git_repository 的 build_file 根 BUILD（havenask 子目录仍用上游自带 BUILD）。
# 勿因「与 rtp_llm 源码无直接 include」删除；havenask 子包（如 aios/autil）依赖 WORKSPACE 内其它 repo（如 @rapidjson）。
#
config_setting(
    name='hack_get_set_env',
    define_values={'hack_get_set_env': 'true'},
    visibility=['//visibility:public']
)

config_setting(
    name = "using_cuda12_arm",
    values = {"define": "using_cuda12_arm=true"},
)

