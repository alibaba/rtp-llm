#ifndef KVCM_KV_META_OBJECT_NATIVE_API_H
#define KVCM_KV_META_OBJECT_NATIVE_API_H

#include <stddef.h>
#include <stdint.h>

// The consumer keeps this capsule alive for the entire call, including while
// the GIL is released. The capsule owns a reference to the existing client;
// normal explicit Close still closes that client and drains admitted calls.
#define KVCM_OBJECT_NATIVE_API_CAPSULE_V2 "kv_cache_manager.ObjectNativeApi.v2"
#define KVCM_OBJECT_NATIVE_API_VERSION_2 2U
#define KVCM_OBJECT_NATIVE_LOAD_CAPSULE_V1 "kv_cache_manager.ObjectNativeLoad.v1"
#define KVCM_OBJECT_NATIVE_LOAD_VERSION_1 1U

typedef struct KvcmObjectNativeSaveRequestV2 {
    const char* key;
    size_t      key_size;
    const char* trace_id;
    size_t      trace_id_size;
    void*       buffer;
    uint64_t    buffer_size;
    uint8_t     memory_type;  // 0 = CPU, 1 = GPU.
    uint8_t     has_userdata1;
    uint8_t     has_userdata2;
    int32_t     userdata1;
    const char* userdata2;
    size_t      userdata2_size;
} KvcmObjectNativeSaveRequestV2;

typedef struct KvcmObjectNativeResultV2 {
    int32_t code;  // ClientErrorCode from the SDK
    uint8_t data_transfer_may_be_inflight;
} KvcmObjectNativeResultV2;

typedef struct KvcmObjectNativeLoadRequestV1 {
    const char* trace_id;
    size_t      trace_id_size;
    void*       buffer;
    uint64_t    buffer_size;
    uint8_t     memory_type;  // 0 = CPU, 1 = GPU.
} KvcmObjectNativeLoadRequestV1;

#ifdef __cplusplus
extern "C" {
#endif

// No callback may throw across the ABI or call Python. The buffer must remain
// alive through the call and longer when its result reports possible DMA.
typedef KvcmObjectNativeResultV2 KvcmObjectNativeSaveV2(void* context, const KvcmObjectNativeSaveRequestV2* request);
typedef KvcmObjectNativeResultV2 KvcmObjectNativeLoadV1(void* context, const KvcmObjectNativeLoadRequestV1* request);

#ifdef __cplusplus
}
#endif

typedef struct KvcmObjectNativeApiV2 {
    uint32_t                version;
    uint32_t                struct_size;
    void*                   context;
    KvcmObjectNativeSaveV2* save;
} KvcmObjectNativeApiV2;

// One read capability for the ObjectMeta returned from the existing Get. No
// Python API or ownership operation is allowed while using the load callback.
// The object remains immutable and its SDK owner remains alive with the capsule.
typedef struct KvcmObjectNativeLoadApiV1 {
    uint32_t                version;
    uint32_t                struct_size;
    void*                   context;
    const void*             client_identity;
    KvcmObjectNativeLoadV1* load;
    uint64_t                object_size;
    uint8_t                 has_userdata1;
    int32_t                 userdata1;
} KvcmObjectNativeLoadApiV1;

#endif
