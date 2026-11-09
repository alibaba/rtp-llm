py_binary(
    name = "package_glm53_extensions",
    srcs = ["build.py"],
    main = "build.py",
    data = glob(["*.cpp", "*.patch"]) + ["check.py", "format", "sources.json"],
    deps = ["//rtp_llm:numpy", "//rtp_llm:setuptools", "//rtp_llm:torch"],
)
