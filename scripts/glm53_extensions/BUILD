py_binary(
    name = "package_glm53_extensions",
    srcs = ["build.py"],
    main = "build.py",
    data = glob(["*.cpp", "*.patch"]) + ["check.py", "sources.json"],
    deps = ["//rtp_llm:torch", "//rtp_llm:setuptools"],
)
