load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")

PACE_TEST_ARTIFACTS = {
    "rpms": {
        "pace_test_meta_rpm": (
            "https://ais-zhouyi-oss1.oss-cn-shenzhen.aliyuncs.com/PACE/release/tair_mempool_metaservice-3.1.21_2026_11_18_16_18-1.al8.x86_64.rpm",
            "4556d4827743178d1799befabc8dc8c917f376dd59ed453085c07377686f20d3",
        ),
        "pace_test_meta_rpm_aarch64": (
            "https://ais-zhouyi-oss1.oss-cn-shenzhen.aliyuncs.com/PACE/release/tair_mempool_metaservice-3.1.21_2026_11_18_16_21-1.al8.aarch64.rpm",
            "84225a39540dfd5cb55d144c4cf6722d213b570dc9d2c0c57fcf8aa64a8fc8f3",
        ),
        "pace_test_server_rpm": (
            "https://ais-zhouyi-oss1.oss-cn-shenzhen.aliyuncs.com/PACE/release/tair_mempool_cssdv_server-3.1.21_2026_11_18_16_24_qp_affinity_mlnx58.9-1.al8.x86_64.rpm",
            "eb2f9657aca1dc61a08dab77e2fac9e16c47885f804d92cf0982332f74a8e7a3",
        ),
        "pace_test_server_rpm_aarch64": (
            "https://ais-zhouyi-oss1.oss-cn-shenzhen.aliyuncs.com/PACE/release/tair_mempool_cssdv_server-3.1.21_2026_11_18_16_26_qp_affinity_mlnx58.9-1.al8.aarch64.rpm",
            "0db40df17c0ff2ed6de8408ac673b9662c69f0f8f6a83023f2ed103ff4e5eced",
        ),
        "pace_test_consumer_rpm": (
            "https://artlab.alibaba-inc.com/1/GENERIC/pace-releases/files/PACE/release/tair_mempool_server-3.1.21_2026_11_20_10_40_50_f14290c5f767_71a5d25e_qp_affinity_mlnx58.9-1.al8.x86_64.rpm?version=3.1.21",
            "b0c111eaff1404205dd4cdd9dcba95085f11919e9cf14dc759c80ea2e09ddc2e",
        ),
        "pace_test_consumer_rpm_aarch64": (
            "https://artlab.alibaba-inc.com/1/GENERIC/pace-releases/files/PACE/release/tair_mempool_server-3.1.21_2026_11_20_10_44_00_f14290c5f767_f94fe727_qp_affinity_mlnx58.9-1.al8.aarch64.rpm?version=3.1.21",
            "d465060856640cf86f5ee751bfc7b524745aa6990dad2f3b4909ea3635a35286",
        ),
        "pace_test_kvcm_sdk_rpm_cuda130_x86": (
            "https://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/kv_cache_manager/client/release/qwen4_dpsk_1116/1dcf928b-ea8b7d8a/kv-cache-manager-client-2026_11_20_16_05-cuda130.x86_64.rpm",
            "93308ad49823ab1e94f2755cbbf6de530c9f2a5e739bcccb8982aff8dffba125",
        ),
        "pace_test_kvcm_sdk_rpm_cuda130_arm": (
            "https://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/kv_cache_manager/client/release/qwen4_dpsk_1116/1dcf928b-ea8b7d8a/kv-cache-manager-client-2026_11_20_16_05-cuda130.aarch64.rpm",
            "97cd811df8bd6b95d9edcc2013131b077d335a8bb35b6db6d9e46f3de5a024b0",
        ),
    },
    "runtime_rpms": {
        "pace_test_runtime_boost_system": (
            "https://mirrors.aliyun.com/alinux/3/updates/x86_64/Packages/boost-system-1.66.0-13.0.3.al8.x86_64.rpm",
            "3946ea7e5192463035f066e9da32282a931e3459146f40e0da270d1f5bad5692",
        ),
        "pace_test_runtime_boost_thread": (
            "https://mirrors.aliyun.com/alinux/3/updates/x86_64/Packages/boost-thread-1.66.0-13.0.3.al8.x86_64.rpm",
            "e13fc3b890ec1ff0b2b608295148816b2756bc64c99457ad5cbcf884dcdc7d66",
        ),
        "pace_test_runtime_gflags": (
            "https://mirrors.aliyun.com/alinux/3/powertools/x86_64/Packages/gflags-2.2.2-1.al8.x86_64.rpm",
            "ba2a711cb9a063df9e03f1ebe2def6b47320f9f056b03b742d1bca3f8c64ce59",
        ),
        "pace_test_runtime_glog": (
            "https://mirrors.aliyun.com/alinux/3/updates/x86_64/Packages/glog-0.3.5-6.0.1.al8.x86_64.rpm",
            "35c4d18f7261f3455a88a20236259a973fc75b842de3be95bc3b84d64dff5f85",
        ),
        "pace_test_runtime_hiredis": (
            "https://mirrors.aliyun.com/epel/8/Everything/x86_64/Packages/h/hiredis-0.13.3-13.el8.x86_64.rpm",
            "7d7aaa0df92c0feedc17b8fac0c6a9a4179d806e211ae7c7cf32ed8796c6954b",
        ),
        "pace_test_runtime_iptables": (
            "https://mirrors.aliyun.com/alinux/3/updates/x86_64/Packages/iptables-libs-1.8.5-11.0.1.al8.x86_64.rpm",
            "c6d2fb84018bbc3657d7963ffa94ac9324e9579194358d71c5e50dcade8b9c36",
        ),
        "pace_test_runtime_jemalloc": (
            "https://mirrors.aliyun.com/epel/8/Everything/x86_64/Packages/j/jemalloc-5.2.1-3.el8.x86_64.rpm",
            "c5d195c29ec8af6bd994b2237984cffdd2b4f9a8ab2aa97c7aad63294f01aa78",
        ),
        "pace_test_runtime_libpcap": (
            "https://mirrors.aliyun.com/alinux/3/os/x86_64/Packages/libpcap-1.9.1-4.1.al8.x86_64.rpm",
            "428ed30b99dc6de006cd57c6bce756d268cfbd225928ddd10b0d90a2a5f6c392",
        ),
        "pace_test_runtime_liburing": (
            "https://mirrors.aliyun.com/alinux/3/updates/x86_64/Packages/liburing-2.3-2.0.1.2.al8.x86_64.rpm",
            "07a830dc3ff203358f79ec81313fedc3c8aa970912d99771869a28d515c60916",
        ),
        "pace_test_runtime_mysql": (
            "https://mirrors.aliyun.com/alinux/3/updates/x86_64/Packages/mysql-libs-8.0.46-1.0.1.1.al8.x86_64.rpm",
            "fe363186dd670c706fc48d78f42c6c869b33c0bc47068d1e3847aeb93a736c2b",
        ),
        "pace_test_runtime_prometheus": (
            "https://mirrors.aliyun.com/epel/8/Everything/x86_64/Packages/l/libprometheus-cpp-1.1.0-1.el8.x86_64.rpm",
            "21d0dec5280816c17e523d8e4a72a365dd917103eca3dbe101630e68796a6380",
        ),
        "pace_test_runtime_tc": (
            "https://mirrors.aliyun.com/alinux/3/updates/x86_64/Packages/iproute-tc-6.2.0-6.0.1.al8.x86_64.rpm",
            "3221f7b12dd7dc1a58582622a664ac78d03d0f6b66d545ce2c86e9ff96dc5b8a",
        ),
    },
    "rpm_layers": {
        "pace_test_meta_rpm": (
            "hub.docker.alibaba-inc.com/system-analysis/tair-mempool-metaservice@sha256:a0ab4b9a4ee2bfe3b879c0b652c2fdddfe8eb55b67ecfeb5bddb7e616e58f40a",
            "tair_mempool/tair_mempool_metaservice-3.1.21-1.al8.x86_64.rpm",
        ),
        "pace_test_server_rpm": (
            "hub.docker.alibaba-inc.com/system-analysis/tair-mempool-server-qp-affinity@sha256:4f98c78baa5afcc74d7e62666476eb59acda6eb5cf20fa8f524d255618c5c204",
            "tair_mempool/tair_mempool_cssdv_server-3.1.21-1.al8.x86_64.rpm",
        ),
    },
    "manager_repo": "hub.docker.alibaba-inc.com/isearch/kv_cache_manager_prod_dashllm",
    "manager_layers": {
        "x86": "a8332a4e6e44110cd4d299071c3c5320a90f736b141454733777f900062c6277",
        "arm": "a309f856fe19744e9781027e02e704ea49d1092cfc96d2b282b3843c2400876d",
    },
}

def pace_test_deps():
    for name, (url, digest) in PACE_TEST_ARTIFACTS["rpms"].items() + PACE_TEST_ARTIFACTS["runtime_rpms"].items():
        http_file(name = name, urls = [url], sha256 = digest)

    registry, image = PACE_TEST_ARTIFACTS["manager_repo"].split("/", 1)
    for architecture, digest in PACE_TEST_ARTIFACTS["manager_layers"].items():
        http_archive(
            name = "pace_test_kvcm_manager_" + architecture,
            urls = [
                "https://%s/v2/%s/blobs/sha256:%s" % (registry, image, digest),
            ],
            sha256 = digest,
            type = "tar",
            strip_prefix = "home/admin/kv_cache_manager",
            patch_cmds = ["tar -xzf kv_cache_manager_server.tar.gz"],
            build_file_content = 'exports_files(["bin/kv_cache_manager_bin", "etc/default_startup_config.json"])',
        )
