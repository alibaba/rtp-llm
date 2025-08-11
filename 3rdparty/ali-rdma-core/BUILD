load("//bazel:defs.bzl", "rpm_library")
 
rpm_library(
    name = "ali-rdma-core" ,
    rpms = ["@ali-rdma-core//file:file"],
    lib_path = "usr/lib64",
    include_path = "usr/include",
    shared_lib = "libfic2.so",
    hdrs = ["infiniband/fic2dv.h"],
)