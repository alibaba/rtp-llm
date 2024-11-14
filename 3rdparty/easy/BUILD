load("//:def.bzl", "rpm_library")

hdrs = [
    "easy/easy_aes.h",
    "easy/easy_aio.h",
    "easy/easy_array.h",
    "easy/easy_atomic.h",
    "easy/easy_baseth_pool.h",
    "easy/easy_buf.h",
    "easy/easy_client.h",
    "easy/easy_connection.h",
    "easy/easy_crc.h",
    "easy/easy_define.h",
    "easy/easy_hash.h",
    "easy/easy_hashx.h",
    "easy/easy_http_handler.h",
    "easy/easy_inet.h",
    "easy/easy_io.h",
    "easy/easy_io_struct.h",
    "easy/easy_list.h",
    "easy/easy_log.h",
    "easy/easy_message.h",
    "easy/easy_pool.h",
    "easy/easy_request.h",
    "easy/easy_simple_handler.h",
    "easy/easy_socket.h",
    "easy/easy_ssl.h",
    "easy/easy_ssl_init.h",
    "easy/easy_string.h",
    "easy/easy_test.h",
    "easy/easy_time.h",
    "easy/easy_timer.h",
    "easy/easy_trace.h",
    "easy/easy_uthread.h",
    "easy/ev.h",
    "easy/http_parser.h",
]

rpm_library(
    name = "easy_headers",
    rpms = [
        "@easy//file:file",
    ],
    hdrs = hdrs,
    include_path = "usr/local/include/",
    lib_path = "",
    include_prefix = ".",
)

rpm_library(
    name = "easy",
    rpms = [
        "@easy//file:file",
    ],
    hdrs = [],
    static_lib = "libeasy.a",
    shared_lib = "libeasy.so",
    include_path = "usr/local/include",
    lib_path = "usr/local/lib",
    deps = [":easy_headers"]
)

