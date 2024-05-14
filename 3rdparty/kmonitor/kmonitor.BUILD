config_setting(
    name='hack_get_set_env',
    define_values={'hack_get_set_env': 'true'},
    visibility=['//visibility:public']
)

cc_library(
    name='kmonitor_client_cpp',
    srcs=glob([
        'aios/kmonitor/cpp_client/src/kmonitor/client/*.cpp',
        'aios/kmonitor/cpp_client/src/kmonitor/client/common/*.cpp',
        'aios/kmonitor/cpp_client/src/kmonitor/client/core/*.cpp',
        'aios/kmonitor/cpp_client/src/kmonitor/client/metric/*.cpp',
        'aios/kmonitor/cpp_client/src/kmonitor/client/net/*.cpp',
        'aios/kmonitor/cpp_client/src/kmonitor/client/net/thrift/*.cpp',
        'aios/kmonitor/cpp_client/src/kmonitor/client/sink/*.cpp'
    ],
              exclude=['aios/kmonitor/cpp_client/src/kmonitor/client/MetricsReporterR.cpp']),
    hdrs=glob([
        'aios/kmonitor/cpp_client/src/kmonitor/client/*.h',
        'aios/kmonitor/cpp_client/src/kmonitor/client/sink/*.h',
        'aios/kmonitor/cpp_client/src/kmonitor/client/core/*.h',
        'aios/kmonitor/cpp_client/src/kmonitor/client/metric/*.h',
        'aios/kmonitor/cpp_client/src/kmonitor/client/net/*.h',
        'aios/kmonitor/cpp_client/src/kmonitor/client/net/thrift/*.h',
        'aios/kmonitor/cpp_client/src/kmonitor/client/common/*.h'
    ],
              exclude=['aios/kmonitor/cpp_client/src/kmonitor/client/MetricsReporterR.h']),
    copts=['-fno-strict-aliasing'],
    implementation_deps=['@havenask//aios/autil:regex'],
    strip_include_prefix='aios/kmonitor/cpp_client/src',
    visibility=['//visibility:public'],
    deps=[
        '@havenask//aios/alog:alog', '@havenask//aios/autil:data_buffer', '@havenask//aios/autil:env_util',
        '@havenask//aios/autil:json', '@havenask//aios/autil:metric', '@havenask//aios/autil:thread',
        '@havenask//aios/network/curl_client:curl_client_lib',
    ],
    alwayslink = True
)

