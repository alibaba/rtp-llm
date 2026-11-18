# c-ares external repository build.
#
# Adapted from grpc@109c570727c3089fef655edcdd0dd02cc5958010
# third_party/cares/cares.BUILD (the same c-ares snapshot is fetched in
# deps/git.bzl). Platform-specific selects are dropped: this repository
# builds the Linux closure only, with config headers vendored under
# rtp_llm/3rdparty/cares and copied in by the two genrules below.
#
# c-ares gives @curl an asynchronous, signal-free resolver
# (CURL_VERSION_ASYNCHDNS): name resolution deadlines are enforced inside
# the transfer timeout loop instead of curl 7.60's synchronous-DNS
# alarm()+siglongjmp path, whose process-directed SIGALRM can be delivered
# to any thread and longjmp across stacks (see OTEL_DNS_HANDOFF.md).

licenses(["notice"])  # c-ares MIT-style license

genrule(
    name = "ares_build_h",
    srcs = ["@rtp_llm//3rdparty/cares:ares_build.h"],
    outs = ["ares_build.h"],
    cmd = "cp $(location @rtp_llm//3rdparty/cares:ares_build.h) $@",
)

genrule(
    name = "ares_config_h",
    srcs = ["@rtp_llm//3rdparty/cares:ares_config_linux.h"],
    outs = ["ares_config.h"],
    cmd = "cp $(location @rtp_llm//3rdparty/cares:ares_config_linux.h) $@",
)

cc_library(
    name = "ares",
    srcs = [
        "ares__close_sockets.c",
        "ares__get_hostent.c",
        "ares__read_line.c",
        "ares__timeval.c",
        "ares_cancel.c",
        "ares_create_query.c",
        "ares_data.c",
        "ares_destroy.c",
        "ares_expand_name.c",
        "ares_expand_string.c",
        "ares_fds.c",
        "ares_free_hostent.c",
        "ares_free_string.c",
        "ares_getenv.c",
        "ares_gethostbyaddr.c",
        "ares_gethostbyname.c",
        "ares_getnameinfo.c",
        "ares_getopt.c",
        "ares_getsock.c",
        "ares_init.c",
        "ares_library_init.c",
        "ares_llist.c",
        "ares_mkquery.c",
        "ares_nowarn.c",
        "ares_options.c",
        "ares_parse_a_reply.c",
        "ares_parse_aaaa_reply.c",
        "ares_parse_mx_reply.c",
        "ares_parse_naptr_reply.c",
        "ares_parse_ns_reply.c",
        "ares_parse_ptr_reply.c",
        "ares_parse_soa_reply.c",
        "ares_parse_srv_reply.c",
        "ares_parse_txt_reply.c",
        "ares_platform.c",
        "ares_process.c",
        "ares_query.c",
        "ares_search.c",
        "ares_send.c",
        "ares_strcasecmp.c",
        "ares_strdup.c",
        "ares_strerror.c",
        "ares_timeout.c",
        "ares_version.c",
        "ares_writev.c",
        "bitncmp.c",
        "inet_net_pton.c",
        "inet_ntop.c",
        "windows_port.c",
    ],
    hdrs = [
        "ares.h",
        "ares_build.h",
        "ares_config.h",
        "ares_data.h",
        "ares_dns.h",
        "ares_getenv.h",
        "ares_getopt.h",
        "ares_inet_net_pton.h",
        "ares_iphlpapi.h",
        "ares_ipv6.h",
        "ares_library_init.h",
        "ares_llist.h",
        "ares_nowarn.h",
        "ares_platform.h",
        "ares_private.h",
        "ares_rules.h",
        "ares_setup.h",
        "ares_strcasecmp.h",
        "ares_strdup.h",
        "ares_version.h",
        "bitncmp.h",
        "config-win32.h",
        "nameser.h",
        "setup_once.h",
    ],
    copts = [
        "-D_GNU_SOURCE",
        "-DHAVE_CONFIG_H",
    ],
    defines = ["CARES_STATICLIB"],
    includes = ["."],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    alwayslink = 1,
)
