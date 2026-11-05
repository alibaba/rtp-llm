package(default_visibility = ["//visibility:public"])

[
    genrule(
        name = "copy_navi_sched_%s" % name,
        srcs = ["//:lib/lib%s.so" % name],
        outs = ["lib%s.so" % name],
        cmd = "cp $< $@",
    )
    for name in ["navi_sched", "lz4", "zstd", "tbb", "elf", "ev"]
]
