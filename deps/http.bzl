load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")
load("@rtp_llm//bazel:tf_http_archive.bzl", "tf_http_archive")

def clean_dep(dep):
    return str(Label(dep))

def http_deps():
    http_archive(
        name = "rules_pkg",
        urls = [
            "http://search-cicd.oss-cn-hangzhou-zmf.aliyuncs.com/third_party_archives/rules_pkg-0.6.0.tar.gz",
            "https://mirror.bazel.build/github.com/bazelbuild/rules_pkg/releases/download/0.6.0/rules_pkg-0.6.0.tar.gz",
            "https://github.com/bazelbuild/rules_pkg/releases/download/0.6.0/rules_pkg-0.6.0.tar.gz",
        ],
        sha256 = "62eeb544ff1ef41d786e329e1536c1d541bb9bcad27ae984d57f18f314018e66",
    )

    http_archive(
        name = "bazel_skylib",
        sha256 = "97e70364e9249702246c0e9444bccdc4b847bed1eb03c5a3ece4f83dfe6abc44",
        urls = ["http://search-cicd.oss-cn-hangzhou-zmf.aliyuncs.com/third_party_archives/bazel-skylib-1.0.2.tar.gz"],
    )

    http_archive(
        name = "io_bazel_rules_closure",
        sha256 = "5b00383d08dd71f28503736db0500b6fb4dda47489ff5fc6bed42557c07c6ba9",
        strip_prefix = "rules_closure-308b05b2419edb5c8ee0471b67a40403df940149",
        urls = [
            "http://pythonrun.cn-hangzhou.oss.aliyun-inc.com/github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",
        ],
    )

    http_archive(
        name = "torch_2.1_py310_cpu",
        sha256 = "bf3ca897f8c7c218dd6c4b1cc5eec57b4f4e71106b0b8120e92f5fdaf4acf6cd",
        urls = [
            "https://download.pytorch.org/whl/cpu/torch-2.1.2%2Bcpu-cp310-cp310-linux_x86_64.whl",
        ],
        type = "zip",
        build_file = clean_dep("//:BUILD.pytorch"),
    )

    http_archive(
        name = "torch_2.6_py310_cuda",
        sha256 = "c55280b4da58e565d8a25e0e844dc27d0c96aaada7b90b4de70a45397faf604e",
        urls = [
            "https://mirrors.aliyun.com/pytorch-wheels/cu126/torch-2.6.0%2Bcu126-cp310-cp310-manylinux_2_28_x86_64.whl",
        ],
        type = "zip",
        build_file = clean_dep("//:BUILD.pytorch"),
    )

    http_archive(
        name = "torch_2.8_py310_cuda",
        sha256 = "37780eb80e4319d6e004ea9597353da0b3947681866d7adff4757ece164a5cd9",
        urls = [
            "https://download.pytorch.org/whl/cu128/torch-2.9.0%2Bcu128-cp310-cp310-manylinux_2_28_aarch64.whl",
        ],
        type = "zip",
        build_file = clean_dep("//:BUILD.pytorch"),
    )

    http_archive(
        name = "torch_2.9_py310_cuda-aarch64",
        sha256 = "37780eb80e4319d6e004ea9597353da0b3947681866d7adff4757ece164a5cd9",
        urls = [
            "https://artlab.alibaba-inc.com/1/PYPI/pytorch/whl/torch/%252Fwhl%252Fcu129/torch-2.9.0%2Bcu129-cp310-cp310-manylinux_2_28_aarch64.whl",
        ],
        type = "zip",
        build_file = clean_dep("//:BUILD.pytorch"),
    )

    http_archive(
        name = "torch_2.11_py310_cuda-aarch64",
        sha256 = "4af01fad0822353e766770ff2c7d6bdc2cbcc2ac7fcd6da93a9e3c6f3f932b21",
        urls = [
            "https://rtp-maga.cn-zhangjiakou.oss.aliyuncs.com/rtp_llm/arm_pkg/torch-2.11.0%2Bcu130-cp310-cp310-manylinux_2_28_aarch64.whl",
        ],
        type = "zip",
        build_file = clean_dep("//:BUILD.pytorch"),
    )

    http_archive(
        name = "torch_2.8_py310_cuda",
        sha256 = "2218bae41c3e4a04ec63b755f78614af18a7ba770b9aa2ed64094c9f6463694a",
        urls = [
            "http://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/zw193905/tmp/torch-2.8.0.dev20250622%2Bcu129-cp310-cp310-manylinux_2_28_x86_64.whl",
        ],
        type = "zip",
        build_file = clean_dep("//:BUILD.pytorch"),
    )

    # CUDA 13.x build of torch 2.11 from upstream pytorch.org/whl/cu130.
    # artlab pytorch mirror does not yet host cu130 wheels, so we point at
    # the OSS mirror the team uploaded (see oss://rtp-maga/miji/0430/).
    http_archive(
        name = "torch_2.11_py310_cuda",
        sha256 = "4c5be01584b7fee22d3c0d04062fd28026044acd07ffd0ee64cbd54b60e62d39",
        urls = [
            "https://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/miji/0430/torch-2.11.0%2Bcu130-cp310-cp310-manylinux_2_28_x86_64.whl",
            "https://download.pytorch.org/whl/cu130/torch-2.11.0%2Bcu130-cp310-cp310-manylinux_2_28_x86_64.whl",
        ],
        type = "zip",
        build_file = clean_dep("//:BUILD.pytorch"),
    )

    http_archive(
        name = "torch_rocm",
        sha256 = "521d1febc9bfebe44fb321727ad550dcaf05900dd917b20bed52fb307f43bf3a",
        urls = [
            "https://sinian-metrics-platform.oss-cn-hangzhou.aliyuncs.com/kis/AMD/torch/torch-2.9.1%2Bgit7e1940d-cp310-cp310-linux_x86_64.whl",
        ],
        type = "zip",
        build_file = clean_dep("//:BUILD.pytorch"),
    )

    http_archive(
        name = "aiter",
        sha256 = "6f0f49ab55490acbce7bb40d147fdeb14418b447d9dfc4b9212dc23ca82b4a88",
        urls = [
            "https://sinian-metrics-platform.oss-cn-hangzhou.aliyuncs.com/kis/AMD/RTP/aiter-0.1.13.dev14%2Bgfa35072d0.d20260402-cp310-cp310-linux_x86_64.whl",
        ],
        type = "zip",
        build_file = clean_dep("//:BUILD.aiter"),
    )

    http_archive(
        name = "torch_2.6_py310_ppu",
        sha256 = "c330a194968849b2aee2001345ce63886513fc47e00301a244ead2944b6def3a",
        urls = [
            "http://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/ppu_sdk/v1.5.2/torch-2.6.0%2Bppu1.5.2.oe-cp310-cp310-linux_x86_64.whl",
        ],
        type = "zip",
        build_file = clean_dep("//internal_source:BUILD.pytorch"),
    )

    http_archive(
        name = "xfastertransformer_devel_icx",
        sha256 = "dfd1714815d38dfea89532365fbe36d502f0bb3baf37c0b472c16743a8cbe352",
        urls = [
            "https://files.pythonhosted.org/packages/b9/0a/9b2da7873a1bada71c3d686ec7ae3a01a1b3864f08b9d07aca7e9e841615/xfastertransformer_devel_icx-1.8.1.1-py3-none-any.whl",
            "https://mirrors.aliyun.com/pypi/packages/b9/0a/9b2da7873a1bada71c3d686ec7ae3a01a1b3864f08b9d07aca7e9e841615/xfastertransformer_devel_icx-1.8.1.1-py3-none-any.whl",
        ],
        type = "zip",
        build_file = clean_dep("//3rdparty/xft:BUILD"),
    )

    http_archive(
        name = "xfastertransformer_devel",
        sha256 = "2344c92cbec175602895bfc76db862a7f724ab9ae0e4aa89bc1b462dfa25b2e9",
        urls = [
            "https://files.pythonhosted.org/packages/a9/67/4133273051133b5848fa29a7da78528c85a013372ee8ca9b90cbc51c4ae0/xfastertransformer_devel-1.8.1.1-py3-none-any.whl",
            "https://mirrors.aliyun.com/pypi/packages/a9/67/4133273051133b5848fa29a7da78528c85a013372ee8ca9b90cbc51c4ae0/xfastertransformer_devel-1.8.1.1-py3-none-any.whl",
        ],
        type = "zip",
        build_file = clean_dep("//3rdparty/xft:BUILD"),
    )

    http_archive(
        name = "torch_2.3_py310_cpu_aarch64",
        sha256 = "bef6996c27d8f6e92ea4e13a772d89611da0e103b48790de78131e308cf73076",
        urls = [
            "https://download.pytorch.org/whl/cpu/torch-2.1.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl#sha256=bef6996c27d8f6e92ea4e13a772d89611da0e103b48790de78131e308cf73076",
        ],
        type = "zip",
        build_file = clean_dep("//:BUILD.pytorch"),
    )

    http_archive(
        name = "arm_compute",
        sha256 = "6d7aebfa9be74d29ecd2dbeb17f69e00c667c36292401f210121bf26a30b38a5",
        urls = ["https://github.com/ARM-software/ComputeLibrary/archive/refs/tags/v24.04.tar.gz"],
        strip_prefix = "ComputeLibrary-24.04",
    )

    http_archive(
        name = "snappy",
        urls = [
            "http://search-cicd.oss-cn-hangzhou-zmf.aliyuncs.com/odps_tensorflow/other/raw/master/mirror.bazel.build/github.com/google/snappy/archive/1.1.7.tar.gz",
            "http://search-cicd.oss-cn-hangzhou-zmf.aliyuncs.com/odps_tensorflow/other/raw/master/github.com/google/snappy/archive/1.1.7.tar.gz",
        ],
        build_file = clean_dep("//3rdparty/snappy:snappy.BUILD"),
        strip_prefix = "snappy-1.1.7",
        sha256 = "3dfa02e873ff51a11ee02b9ca391807f0c8ea0529a4924afa645fbf97163f9d4",
    )

    http_file(
        name = "zookeeper-package",
        urls = ["http://search-cicd.oss-cn-hangzhou-zmf.aliyuncs.com/third_party_archives/zookeeper_c_client-3.4.14-rc_5.x86_64.rpm"],
        sha256 = "5a604258ef72438e35c0cc5f039189e10168e074c392df183fbe522f537bc046",
    )

    http_file(
        name = "mxml-package",
        urls = ["http://yum.tbsite.net/taobao/7/x86_64/current/mxml/mxml-2.6-1.alios7.x86_64.rpm"],
        sha256 = "383985a3f60bdefc0ef016d7c25fc84b6ed182f5a356d4432d331cc902687987",
    )

    tf_http_archive(
        name = "jsoncpp_git",
        build_file = clean_dep("//3rdparty/jsoncpp:jsoncpp.BUILD"),
        sha256 = "c49deac9e0933bcb7044f08516861a2d560988540b23de2ac1ad443b219afdb6",
        strip_prefix = "jsoncpp-1.8.4",
        system_build_file = clean_dep("//3rdparty/jsoncpp:jsoncpp-systemlib.BUILD"),
        urls = [
            "http://search-cicd.oss-cn-hangzhou-zmf.aliyuncs.com/odps_tensorflow/other/raw/master/mirror.bazel.build/github.com/open-source-parsers/jsoncpp/archive/1.8.4.tar.gz",
            "http://search-cicd.oss-cn-hangzhou-zmf.aliyuncs.com/odps_tensorflow/other/raw/master/github.com/open-source-parsers/jsoncpp/archive/1.8.4.tar.gz",
        ],
    )

    http_archive(
        name = "boost",
        urls = [
            "http://search-cicd.oss-cn-hangzhou-zmf.aliyuncs.com/third_party_archives/boost_1_70_0.tar.gz",
            "https://boostorg.jfrog.io/artifactory/main/release/1.70.0/source/boost_1_70_0.tar.gz",
        ],
        build_file = clean_dep("//3rdparty/boost:boost.BUILD"),
        # https://github.com/boostorg/hana/issues/446
        patches = ["//patches/boost:boost.patch"],
        strip_prefix = "boost_1_70_0",
        sha256 = "882b48708d211a5f48e60b0124cf5863c1534cd544ecd0664bb534a4b5d506e9",
    )

    http_file(
        name = "easy",
        urls = [
            "http://yum.tbsite.net/taobao/7/x86_64/current/t_libeasy/t_libeasy-1.1.33-799806.el7.x86_64.rpm",
        ],
        sha256 = "2509b359cb0d784dee719c214c491aa5c9c75f623807a0083499adbc7334c3c0",
    )

    http_file(
        name = "solar",
        urls = ["https://search-cicd.oss-cn-hangzhou-zmf.aliyuncs.com/third_party_archives/solar-1.0.0-1.x86_64.rpm"],
        sha256 = "12781d5301e641e6011d33949e59058e987068a7bf28ce2966fe1e99e9a4bde0",
    )

    http_file(
        name = "tnet",
        urls = ["https://search-cicd.oss-cn-hangzhou-zmf.aliyuncs.com/third_party_archives/tnet-devel-3.1.0-1.noarch.rpm"],
        sha256 = "5d78de3c3bd15b2e66448470b9d01006a2e45cfc1b5858f0b180f9f7c588882e",
    )

    http_file(
        name = "unicm",
        urls = ["http://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/eic-packages%2Funicm-1.8.1-1.x86_64.rpm"],
        sha256 = "654d707399a7c40a159ed5e5e9aa969fa678c8333b62fdb8da6101c73daecbef",
    )

    http_file(
        name = "u2mm",
        urls = ["http://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/eic-packages/u2mm-3.3.2-r20250303_cuda11.x86_64.rpm"],
        sha256 = "816500b86207b178893c1a9f638249a3f2ba2493c1c5de2036851bc65b6cdb2c",
    )

    http_file(
        name = "ali-rdma-core",
        urls = ["http://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/eic-packages/ali-rdma-core-2506.1-1.x86_64.rpm"],
        sha256 = "9ba4080a8ca9ba8d8632e488b61928aa2355ae36c1a6027a097818be9bfc0253",
    )

    http_file(
        name = "accl_ep_rpm",
        urls = ["http://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/pkg%2Facclep%2FAcclEP-1.1.0.9-f02e709c.alios7.x86_64.rpm"],
        sha256 = "553ef1ad72f4e52d679c84b1795ee44e4a0571aa4c7fb0d3c362be01bd54c807",
    )

    http_archive(
        # Hedron's Compile Commands Extractor for Bazel
        name = "hedron_compile_commands",
        urls = ["https://github.com/hedronvision/bazel-compile-commands-extractor/archive/4f28899228fb3ad0126897876f147ca15026151e.tar.gz"],
        strip_prefix = "bazel-compile-commands-extractor-4f28899228fb3ad0126897876f147ca15026151e",
        sha256 = "658122cfb1f25be76ea212b00f5eb047d8e2adc8bcf923b918461f2b1e37cdf2",
    )

    http_file(
        name = "hf3fs_rpm",
        urls = ["http://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/pkg%2F3fs%2Fhf3fs-1.3.0-1.alios7.x86_64.rpm"],
        sha256 = "dd375f794557a1135934b40b23a7435569644922c5c7116cb69dd36f699ad5a4",
    )

    http_file(
        name = "remote_kv_cache_manager_client_rpm",
        urls = [
            "http://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/kv_cache_manager/client/kv-cache-manager-client-2026_08_27_17_09.rpm",
        ],
        sha256 = "2a548641d5dd0b552657524c65d96e3c01255143b8418f3d953a9d4e9446a258",
    )

    http_archive(
        name = "remote_kv_cache_manager_server",
        urls = [
            "http://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/kv_cache_manager/server/kv_cache_manager_server_2026_08_19_20_01.tar.gz",
        ],
        sha256 = "facbcee3395e3fa129cf45c7034d82e576fefc5af7017fead26ed795bb6f7bf6",
        build_file_content = """
exports_files(["bin/kv_cache_manager_bin"])
        """,
    )
