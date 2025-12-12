def java_junit5_test(
        name,
        srcs,
        test_package,
        deps = [],
        runtime_deps = [],
        deps_name = "maven",
        **kwargs):
    FILTER_KWARGS = [
        "main_class",
        "use_testrunner",
        "args",
    ]

    for arg in FILTER_KWARGS:
        if arg in kwargs.keys():
            kwargs.pop(arg)

    junit_console_args = []
    if test_package:
        junit_console_args += ["--select-package", test_package, "-n", ".*"]
    else:
        fail("must specify 'test_package'")

    use_testrunner = False
    main_class = "org.junit.platform.console.ConsoleLauncher"
    jvm_flags = kwargs.pop("jvm_flags", [])
    deps = deps + [
        "@" + deps_name + "//:org_junit_jupiter_junit_jupiter_api",
        "@" + deps_name + "//:org_junit_jupiter_junit_jupiter_params",
        "@" + deps_name + "//:org_junit_jupiter_junit_jupiter_engine",
    ]
    runtime_deps = runtime_deps + [
        "@" + deps_name + "//:org_junit_platform_junit_platform_console",
    ]

    native.java_test(
        name = name,
        srcs = srcs,
        use_testrunner = use_testrunner,
        main_class = main_class,
        args = junit_console_args,
        jvm_flags = ["-ea"] + jvm_flags,
        deps = deps,
        runtime_deps = runtime_deps,
        **kwargs
    )

    native.java_test(
        name = name + "_release",
        srcs = srcs,
        use_testrunner = use_testrunner,
        main_class = main_class,
        args = junit_console_args,
        jvm_flags = ["-da"] + jvm_flags,
        deps = deps,
        runtime_deps = runtime_deps,
        **kwargs
    )
