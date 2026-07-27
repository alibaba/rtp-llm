#pragma once

#include <stdint.h>

#if !defined(__BYTE_ORDER__) || __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "multicast keeper protocol V3 requires little-endian wire encoding"
#endif

#define RTP_MC_PROTOCOL_MAGIC 0x3250434d505452ull /* "RTPMCP2" */
#define RTP_MC_CREATOR_MAGIC 0x3243464d505452ull  /* "RTPMCF2" */
#define RTP_MC_PROTOCOL_VERSION 3
#define RTP_MC_DEFAULT_SOCKET_NAME "mcsk.sock"
#define RTP_MC_MAX_ENTRIES 256
#define RTP_MC_TOKEN_MAGIC "RTPMCTK3"

#define RTP_MC_HANDLE_TYPE_POSIX_FD 0x1u
#define RTP_MC_HANDLE_TYPE_FABRIC 0x8u
#define RTP_MC_SUPPORTED_HANDLE_TYPES (RTP_MC_HANDLE_TYPE_POSIX_FD | RTP_MC_HANDLE_TYPE_FABRIC)

// A CUDA CU_MEM_HANDLE_TYPE_FABRIC shareable handle is 64 opaque bytes. This is
// the unit of cross-machine (MNNVL / NVL72) exchange: unlike a POSIX fd it can
// be memcpy'd, broadcast over the c10d store, and imported on any node in the
// NVLink fabric. Node-local ranks still import via a POSIX fd (see below).
#define RTP_MC_FABRIC_HANDLE_BYTES 64
// IMPORT_ADD uses this explicit sentinel when the peer CUDA API exposes only an
// opaque raw handle. A raw-handle dedup hit adopts the holder's known size.
#define RTP_MC_UNKNOWN_SIZE UINT64_MAX

typedef enum rtp_mc_opcode {
    RTP_MC_OP_PING   = 1,
    RTP_MC_OP_CREATE = 2,
    RTP_MC_OP_FETCH  = 3,
    // Release this process incarnation's reference to an object. The holder
    // closes the object only after its last registered owner releases it.
    RTP_MC_OP_RELEASE = 4,
    // Cross-machine fabric ops (single-node POSIX path never emits these):
    //   IMPORT_ADD   — a peer-node rank hands its local holder a 64-byte fabric
    //                  handle (produced by the creator node and broadcast over
    //                  the store). The holder imports it, AddDevice's its LOCAL
    //                  devices, re-exports a node-local POSIX fd, and holds it —
    //                  after which local ranks FETCH exactly as single-node.
    //   FETCH_FABRIC — the creating rank retrieves the 64-byte fabric handle so
    //                  it can publish it to peers. The handle is returned in a
    //                  sealed POSIX memfd over SCM_RIGHTS (the response struct is
    //                  unchanged), mirroring how identity tokens are transported.
    RTP_MC_OP_IMPORT_ADD   = 5,
    RTP_MC_OP_FETCH_FABRIC = 6,
} rtp_mc_opcode;

typedef enum rtp_mc_status {
    RTP_MC_STATUS_OK                     = 0,
    RTP_MC_STATUS_INVALID_REQUEST        = 1,
    RTP_MC_STATUS_UNSUPPORTED_PROPERTIES = 2,
    RTP_MC_STATUS_STALE_INSTANCE         = 3,
    RTP_MC_STATUS_UNKNOWN_OBJECT         = 4,
    RTP_MC_STATUS_PROPERTY_MISMATCH      = 5,
    RTP_MC_STATUS_CREATOR_FAILED         = 6,
    RTP_MC_STATUS_CAPACITY_EXCEEDED      = 7,
    RTP_MC_STATUS_INTERNAL_ERROR         = 8,
    // RELEASE (or an owner-scoped operation) was refused because the exact
    // (owner_id, owner_generation) pair is not registered. Fail closed.
    RTP_MC_STATUS_OWNER_MISMATCH = 9,
} rtp_mc_status;

typedef struct rtp_mc_object_properties {
    uint64_t size;
    uint32_t num_devices;
    uint32_t handle_types;
    uint64_t flags;
} rtp_mc_object_properties;

// Little-endian wire layout: <QHHIQQQQIIQ (64 bytes). CREATE requires a zero
// holder instance/object id. FETCH requires the exact identity from CREATE.
typedef struct rtp_mc_request {
    uint64_t                 magic;
    uint16_t                 version;
    uint16_t                 opcode;
    uint32_t                 struct_size;
    uint64_t                 holder_instance_hi;
    uint64_t                 holder_instance_lo;
    uint64_t                 object_id;
    rtp_mc_object_properties properties;
} rtp_mc_request;

// Owner attribution appended after the 64-byte base request. A sender that
// provides it sets base.struct_size == sizeof(rtp_mc_request_ext) and transmits
// the full 80-byte SEQPACKET message; the holder distinguishes the two forms by
// message length. A base (64-byte) request implies owner_id == 0.
//
// Ownership model:
//   - CREATE registers its caller. IMPORT_ADD registers each distinct peer
//     process incarnation that imports the raw FABRIC handle. Registration is
//     idempotent for an exact (owner_id, owner_generation) pair.
//   - owner_id is a logical, restart-stable owner key (e.g. a rank slot).
//   - owner_generation is a per-incarnation nonce that changes whenever the
//     owning process is relaunched (stable across a checkpoint/restore cycle).
//   - CREATE and IMPORT_ADD carrying a nonzero owner_id remove stale references
//     of that owner_id with a different generation. The entry remains alive for
//     other owners and closes only when its owner set becomes empty.
//   - RELEASE is honored only when (holder_instance, object_id) resolve to an
//     entry containing the exact owner pair; otherwise it fails closed.
// Little-endian wire layout: <QHHIQQQQIIQQQ (80 bytes).
typedef struct rtp_mc_request_ext {
    rtp_mc_request base;
    uint64_t       owner_id;
    uint64_t       owner_generation;
} rtp_mc_request_ext;

// Little-endian wire layout: <QHHIiIQQQQQIIQ (80 bytes).
typedef struct rtp_mc_response {
    uint64_t magic;
    uint16_t version;
    uint16_t opcode;
    uint32_t struct_size;
    int32_t  status;
    // Exact number of local devices configured on this holder (V3+).
    uint32_t local_device_count;
    uint64_t holder_instance_hi;
    uint64_t holder_instance_lo;
    uint64_t object_id;
    uint64_t requested_size;
    uint64_t served_size;
    uint32_t num_devices;
    uint32_t handle_types;
    uint64_t flags;
} rtp_mc_response;

// The CUDA FABRIC handle ABI is 64 bytes. Keep the token exactly that size so
// the same authenticated object reference can be transported inline or in a
// sealed POSIX memfd.
typedef struct rtp_mc_token {
    char                     magic[8];
    uint16_t                 version;
    uint16_t                 token_size;
    uint32_t                 reserved;
    uint64_t                 holder_instance_hi;
    uint64_t                 holder_instance_lo;
    uint64_t                 object_id;
    rtp_mc_object_properties properties;
} rtp_mc_token;

// Creator-to-holder transfer uses a private SOCK_SEQPACKET socketpair. The
// node-local multicast object always travels as a POSIX fd over SCM_RIGHTS; the
// FABRIC handle (when the object was created FABRIC|POSIX for cross-node import)
// travels inline in fabric_handle, flagged by RTP_MC_CREATOR_FLAG_FABRIC_VALID.
// A peer-node importer child reuses this same result to return its re-exported
// POSIX fd, leaving fabric_handle unset (single-node POSIX path leaves it unset
// too, so flags == 0 and the extra bytes are ignored).
#define RTP_MC_CREATOR_FLAG_FABRIC_VALID 0x1u

typedef struct rtp_mc_creator_result {
    uint64_t      magic;
    uint64_t      requested_size;
    uint64_t      served_size;
    int32_t       status;
    uint32_t      flags;
    unsigned char fabric_handle[RTP_MC_FABRIC_HANDLE_BYTES];
} rtp_mc_creator_result;

// IMPORT_ADD request wire form: the 80-byte extended request (opcode
// RTP_MC_OP_IMPORT_ADD) immediately followed by the 64 raw fabric-handle bytes,
// for a total of 144 bytes. The holder distinguishes IMPORT_ADD from CREATE /
// FETCH / RELEASE by opcode and by the 144-byte SEQPACKET message length.
typedef struct rtp_mc_import_add_request {
    rtp_mc_request_ext ext;
    unsigned char      fabric_handle[RTP_MC_FABRIC_HANDLE_BYTES];
} rtp_mc_import_add_request;

#if defined(__cplusplus)
static_assert(sizeof(rtp_mc_object_properties) == 24, "property layout changed");
static_assert(sizeof(rtp_mc_request) == 64, "request layout changed");
static_assert(sizeof(rtp_mc_request_ext) == 80, "extended request layout changed");
static_assert(sizeof(rtp_mc_response) == 80, "response layout changed");
static_assert(sizeof(rtp_mc_token) == 64, "token layout changed");
static_assert(sizeof(rtp_mc_creator_result) == 96, "creator layout changed");
static_assert(sizeof(rtp_mc_import_add_request) == 144, "import-add layout changed");
#else
_Static_assert(sizeof(rtp_mc_object_properties) == 24, "property layout changed");
_Static_assert(sizeof(rtp_mc_request) == 64, "request layout changed");
_Static_assert(sizeof(rtp_mc_request_ext) == 80, "extended request layout changed");
_Static_assert(sizeof(rtp_mc_response) == 80, "response layout changed");
_Static_assert(sizeof(rtp_mc_token) == 64, "token layout changed");
_Static_assert(sizeof(rtp_mc_creator_result) == 96, "creator layout changed");
_Static_assert(sizeof(rtp_mc_import_add_request) == 144, "import-add layout changed");
#endif
