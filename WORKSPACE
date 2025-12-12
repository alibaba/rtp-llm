workspace(name = "rtp_llm")

load("//3rdparty/cuda_config:cuda_configure.bzl", "cuda_configure")
load("//3rdparty/gpus:rocm_configure.bzl", "rocm_configure")
load("//3rdparty/py:python_configure.bzl", "python_configure")

cuda_configure(name = "local_config_cuda")

rocm_configure(name = "local_config_rocm")

python_configure(name = "local_config_python")

load("//deps:http.bzl", "http_deps")

http_deps()

load("//deps:git.bzl", "git_deps")

git_deps()

load("@rules_python//python:repositories.bzl", "py_repositories")

py_repositories()

load("//deps:pip.bzl", "pip_deps")

pip_deps()

load("@pip_cpu_torch//:requirements.bzl", pip_cpu_torch_install_deps = "install_deps")
pip_cpu_torch_install_deps()

load("@pip_arm_torch//:requirements.bzl", pip_arm_torch_install_deps = "install_deps")
pip_arm_torch_install_deps()

load("@pip_ppu_torch//:requirements.bzl", pip_ppu_torch_install_deps = "install_deps")
pip_ppu_torch_install_deps()

load("@pip_gpu_cuda12_torch//:requirements.bzl", pip_gpu_cuda12_torch_install_deps = "install_deps")
pip_gpu_cuda12_torch_install_deps()

load("@pip_gpu_cuda12_9_torch//:requirements.bzl", pip_gpu_cuda12_9_torch_install_deps = "install_deps")
pip_gpu_cuda12_9_torch_install_deps()

load("@pip_cuda12_arm_torch//:requirements.bzl", pip_cuda12_arm_torch_install_deps = "install_deps")
pip_cuda12_arm_torch_install_deps()

load("@pip_gpu_rocm_torch//:requirements.bzl", pip_gpu_rocm_torch_install_deps = "install_deps")
pip_gpu_rocm_torch_install_deps()

load("//:def.bzl", "read_release_version")
read_release_version(name = "release_version")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "rules_jvm_external",
    sha256 = "d31e369b854322ca5098ea12c69d7175ded971435e55c18dd9dd5f29cc5249ac",
    strip_prefix = "rules_jvm_external-5.3",
    url = "https://github.com/bazelbuild/rules_jvm_external/releases/download/5.3/rules_jvm_external-5.3.tar.gz",
)

load("@rules_jvm_external//:repositories.bzl", "rules_jvm_external_deps")
rules_jvm_external_deps()

load("@rules_jvm_external//:setup.bzl", "rules_jvm_external_setup")
rules_jvm_external_setup()

# Java rules
http_archive(
    name = "rules_java",
    urls = [
        "https://github.com/bazelbuild/rules_java/releases/download/7.1.0/rules_java-7.1.0.tar.gz",
    ],
    sha256 = "a37a4e5f63ab82716e5dd6aeef988ed8461c7a00b8e936272262899f587cd4e1",
)
load("@rules_java//java:repositories.bzl", "rules_java_dependencies", "rules_java_toolchains")
rules_java_dependencies()
rules_java_toolchains()


# Load rules_proto
http_archive(
    name = "rules_proto",
    sha256 = "dc3fb206a2cb3441b485eb1e423165b231235a1ea9b031b4433cf7bc1fa460dd",
    strip_prefix = "rules_proto-5.3.0-21.7",
    urls = [
        "https://github.com/bazelbuild/rules_proto/archive/refs/tags/5.3.0-21.7.tar.gz",
    ],
)

load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies", "rules_proto_toolchains")
rules_proto_dependencies()
rules_proto_toolchains()


#For released versions, use release tag:
http_archive(
    name = "io_grpc_grpc_java",
    sha256 = "59ded3553cf6f5d6ecc26eccc22cb267692af67ac73520aed5105faf60ce34b5",
    strip_prefix = "grpc-java-1.65.1",
    urls = [
        "https://github.com/grpc/grpc-java/archive/v1.65.1.tar.gz",
    ],
)

load("@rules_jvm_external//:defs.bzl", "maven_install")
load("@io_grpc_grpc_java//:repositories.bzl", "IO_GRPC_GRPC_JAVA_ARTIFACTS")
load("@io_grpc_grpc_java//:repositories.bzl", "IO_GRPC_GRPC_JAVA_OVERRIDE_TARGETS")
load("@io_grpc_grpc_java//:repositories.bzl", "grpc_java_repositories")

grpc_java_repositories()

# Protobuf now requires C++14 or higher, which requires Bazel configuration
# outside the WORKSPACE. See .bazelrc in this directory.
#load("@com_google_protobuf//:protobuf_deps.bzl", "PROTOBUF_MAVEN_ARTIFACTS")
# load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

# protobuf_deps()
http_archive(
    name = "com_google_protobuf",
    sha256 = "1a2affa2fbad568b9895b72e3c7cb1f72a14bf2501fba056c724dc68c249cd0f",
    strip_prefix = "protobuf-3.25.1",
    urls = ["https://github.com/protocolbuffers/protobuf/archive/v3.25.1.tar.gz"],
)
load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps", "PROTOBUF_MAVEN_ARTIFACTS")
protobuf_deps()

# PROTOBUF_MAVEN_ARTIFACTS = [
#     "com.google.code.findbugs:jsr305:3.0.2",
#     "com.google.code.gson:gson:2.8.9",
#     "com.google.errorprone:error_prone_annotations:2.5.1",
#     "com.google.j2objc:j2objc-annotations:2.8",
#     "com.google.guava:guava:32.0.1-jre",
#     "com.google.protobuf:protobuf-java:3.25.1",
# ]

maven_install(
    artifacts = [
        "com.google.api.grpc:grpc-google-cloud-pubsub-v1:0.1.24",
        "com.google.api.grpc:proto-google-cloud-pubsub-v1:0.1.24",
    ] + IO_GRPC_GRPC_JAVA_ARTIFACTS + PROTOBUF_MAVEN_ARTIFACTS,
    generate_compat_repositories = True,
    override_targets = IO_GRPC_GRPC_JAVA_OVERRIDE_TARGETS,
    repositories = [
        "https://repo.maven.apache.org/maven2/",
    ],
)

load("@maven//:compat.bzl", "compat_repositories")

compat_repositories()

#Call the function to set up repositories


# flexlb_deps()

# # Now load and set up the dependencies
# load("@rules_jvm_external//:repositories.bzl", "rules_jvm_external_deps")
# rules_jvm_external_deps()

# load("@rules_jvm_external//:setup.bzl", "rules_jvm_external_setup")
# rules_jvm_external_setup()

# load("@rules_java//java:repositories.bzl", "rules_java_dependencies", "rules_java_toolchains")
# rules_java_dependencies()
# rules_java_toolchains()

# load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies", "rules_proto_toolchains")
# rules_proto_dependencies()
# rules_proto_toolchains()

# load("@io_grpc_grpc_java//:repositories.bzl", "IO_GRPC_GRPC_JAVA_ARTIFACTS", "IO_GRPC_GRPC_JAVA_OVERRIDE_TARGETS", "grpc_java_repositories")
# grpc_java_repositories()

# load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps", "PROTOBUF_MAVEN_ARTIFACTS")
# protobuf_deps()

# load("@rules_jvm_external//:defs.bzl", "maven_install")

# # Get Maven configuration
# maven_config = flexlb_maven_init()

# maven_install(
#     artifacts = maven_config["artifacts"] + IO_GRPC_GRPC_JAVA_ARTIFACTS + PROTOBUF_MAVEN_ARTIFACTS,
#     generate_compat_repositories = True,
#     override_targets = IO_GRPC_GRPC_JAVA_OVERRIDE_TARGETS,
#     repositories = maven_config["repositories"],
# )

# load("@maven//:compat.bzl", "compat_repositories")
# compat_repositories()

load("//rtp_llm/flexlb:deps.bzl", "flexlb_init")
flexlb_maven_config = flexlb_init()
maven_install(
    name = flexlb_maven_config["name"],
    artifacts = flexlb_maven_config["artifacts"],
    generate_compat_repositories = True,
    override_targets = IO_GRPC_GRPC_JAVA_OVERRIDE_TARGETS,
    repositories = flexlb_maven_config["repositories"],
    strict_visibility = True,
)