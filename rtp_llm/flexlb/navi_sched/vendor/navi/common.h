#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/atomic.h"
#include "autil/CommonMacros.h"
#include "navi/util/NaviTestPool.h"
#include <algorithm>
#include <assert.h>
#include <functional>
#include <memory>
#include <stdint.h>

#define USE_ABSL_CONTAINER

#ifdef USE_ABSL_CONTAINER
#include "absl/container/inlined_vector.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"

template<typename K, typename V, typename hash = absl::container_internal::hash_default_hash<K>>
using NaviUnorderedMap = absl::flat_hash_map<K, V, hash>;

template<typename K>
using NaviUnorderedSet =  absl::flat_hash_set<K>;

template<typename T, size_t N>
using NaviVector = absl::InlinedVector<T, N>;

namespace autil {
template<typename K, typename V, typename hash>
inline std::ostream &operator<<(std::ostream &os, const NaviUnorderedMap<K, V, hash> &x) {
    for (auto iter = x.begin(); iter != x.end(); ++iter) {
        os << iter->first << ":" << iter->second << ";";
    }
    return os;
}
}

#else
#include <unordered_map>
#include <unordered_set>
#include <vector>

template<typename K, typename V, typename hash = std::hash<K>>
using NaviUnorderedMap = std::unordered_map<K, V, hash>;

template<typename K>
using NaviUnorderedSet =  std::unordered_set<K>;

template<typename T, size_t N>
using NaviVector = std::vector<T>;
#endif

namespace navi {

#define NAVI_TYPEDEF_PTR(x)                                                                                            \
    typedef std::shared_ptr<x> x##Ptr;                                                                                 \
    typedef std::unique_ptr<x> x##UPtr;

enum ErrorCode : int32_t {
    EC_NONE = 0,
    EC_ABORT = 1,
    EC_DEADLOCK = 2,
    EC_INVALID_ATTRIBUTE = 3,
    EC_INIT_GRAPH = 4,
    EC_INIT_SUB_GRAPH = 5,
    EC_WEAVE_GRAPH = 6,
    EC_QUEUE_FULL = 7,
    EC_FORK = 8,
    EC_TIMEOUT = 9,
    EC_INIT_RESOURCE = 10,
    EC_CREATE_KERNEL = 11,
    EC_LACK_RESOURCE = 12,
    EC_RPC_INPUT_PORT = 13,
    EC_PROTO_DEF = 14,
    EC_DESERIALIZE = 15,
    EC_SERIALIZE = 16,
    EC_NO_DATA = 17,
    EC_DATA_OVERRIDE = 18,
    EC_DATA_AFTER_EOF = 19,
    EC_BUFFER_FULL = 20,
    EC_STREAM_RPC = 21,
    EC_STREAM_RECEIVE = 22,
    EC_STREAM_SEND = 23,
    EC_NO_GRAPH_OUTPUT = 24,
    EC_OVERRIDE_EDGE_DATA = 25,
    EC_KERNEL = 32,
    EC_PART_ID_OVERFLOW = 33,
    EC_ASYNC_TERMINATED = 34,
    EC_PART_CANCEL = 35,
    EC_PART_ID_NOT_EXIST = 36,
    EC_PART_COUNT_MISMATCH = 37,
    EC_LACK_REGISTRY = 38,
    EC_ILLEGAL_RESOURCE_STAGE = 39,
    EC_RESOURCE_CONFIG = 40,
    EC_BIND_RESOURCE = 41,
    EC_BIND_NAMED_DATA = 42,
    EC_LACK_RESOURCE_SNAPSHOT = 43,
    EC_LACK_RESOURCE_BIZ = 44,
    EC_LACK_RESOURCE_BIZ_PART = 45,
    EC_LACK_RESOURCE_GRAPH = 46,
    EC_LACK_RESOURCE_SUB_GRAPH = 47,
    EC_LACK_RESOURCE_SCOPE = 48,
    EC_LACK_RESOURCE_KERNEL = 49,
    EC_LACK_RESOURCE_EXTERNAL = 50,
    EC_LACK_REPLACE_RESOURCE = 51,
    EC_REWRITE_GRAPH = 52,
    EC_ADD_NAMED_DATA = 53,
    EC_NEED_REPLACE = 54,
    EC_SERIALIZE_OVERRIDE_DATA = 55,
    EC_SERIALIZE_NAMED_DATA = 56,
    EC_ARPC_RESULT = 57,
    EC_FATAL_ABORT = 58,
    EC_UNKNOWN = 127,
    EC_NAVI_MAX,
};

typedef int32_t NaviPartId;
constexpr NaviPartId INVALID_NAVI_PART_COUNT = 0;
constexpr NaviPartId INVALID_NAVI_PART_ID = -1;
constexpr NaviPartId NAVI_BIZ_PART_ID = -2;

enum TypeErrorCode {
    TEC_NONE,
    TEC_NOT_SUPPORT,
    TEC_NULL_DATA,
    TEC_TYPE_MISMATCH,
    TEC_FAILED,
    TEC_UNKNOWN,
};

enum SummaryErrorCode {
    SEC_OK = 0,
    SEC_IGNORE_DATA = 1,
    SEC_FAILED = 2,
    SEC_NULL_DATA = 3,
    SEC_NOT_SUPPORT = 4,
};

enum ActivateStrategy {
    AS_REQUIRED = 0,
    AS_OPTIONAL = 1,
    AS_INDEPENDENT = 2
};

enum IoType {
    IOT_INPUT = 0,
    IOT_OUTPUT = 1,
};

enum PortStoreType {
    PST_DATA = 0,
    PST_SERIALIZE_DATA = 1,
    PST_UNKNOWN = 8,
};

enum PortQueueType {
    PQT_LOCAL_REMOTE_SERVER_1 = 0,
    PQT_LOCAL_REMOTE_CLIENT_2 = 1,
    PQT_LOCAL_LOCAL_3 = 2,
    PQT_REMOTE_LOCAL_SERVER_4 = 3,
    PQT_REMOTE_LOCAL_CLIENT_5 = 4,
    PQT_REMOTE_REMOTE_CLIENT_6 = 5,
    PQT_UNKNOWN = 32
};

enum EdgeGetType {
    EGT_GET = 0,
    EGT_CLEAR = 1,
    EGT_PEEK = 2,
};

typedef uint32_t InstanceId;
typedef uint32_t QueryId;
struct SessionId {
    SessionId()
        : instance(0)
        , currentInstance(0)
        , queryId(0)
    {
    }
    bool valid() const {
        return 0 != instance && 0 != queryId;
    }
    InstanceId instance;
    InstanceId currentInstance;
    QueryId queryId;
};

typedef uint16_t IndexType;
constexpr IndexType INVALID_INDEX = IndexType(-1);

typedef uint32_t PortId;
constexpr PortId INVALID_PORT_ID = PortId(-1);

struct PortIndex
{
private:
    static constexpr IndexType CONTROL_MASK = (1u << 15);
public:
    PortIndex(IndexType index = INVALID_INDEX, IndexType group = INVALID_INDEX, bool control = false)
        : _index(index)
        , _group(group & ~(CONTROL_MASK))
    {
        if (control) {
            _group = (_group | CONTROL_MASK);
        }
    }
    bool isValid() const {
        return !(*this == PortIndex());
    }
    bool operator==(const PortIndex &rhs) const {
        return _index == rhs._index
            && _group == rhs._group;
    }
    bool operator!=(const PortIndex &rhs) const {
        return !(*this == rhs);
    }
    void setIndex(IndexType index) {
        _index = index;
    }
    void setGroup(IndexType group) {
        if (isControl()) {
            _group = (group | CONTROL_MASK);
        } else {
            _group = group;
        }
    }
    IndexType index() const {
        return _index;
    }
    IndexType group() const {
        return _group & (~CONTROL_MASK);
    }
    bool isGroup() const {
        return group() != (INVALID_INDEX  & (~CONTROL_MASK));
    }
    bool isControl() const {
        return (_group & CONTROL_MASK);
    }
private:
    IndexType _index;
    IndexType _group;
};

extern std::ostream &operator<<(std::ostream &os, const PortIndex &portIndex);

enum LogLevel {
    LOG_LEVEL_DISABLE = 0,
/** the fatal message level*/
    LOG_LEVEL_FATAL = 1,
/** the error message level*/
    LOG_LEVEL_ERROR = 2,
/** the warning message level*/
    LOG_LEVEL_WARN = 3,
/** the information level*/
    LOG_LEVEL_INFO = 4,
/** the debug message level*/
    LOG_LEVEL_DEBUG = 5,
/** the trace level 1 */
    LOG_LEVEL_TRACE1 = 6,
/** the trace level 2 */
    LOG_LEVEL_TRACE2 = 7,
/** the trace level 3 */
    LOG_LEVEL_TRACE3 = 8,
/** the schedule info level*/
    LOG_LEVEL_SCHEDULE1 = 9,
    LOG_LEVEL_SCHEDULE2 = 10,
    LOG_LEVEL_SCHEDULE3 = 11,
/** level not set */
    LOG_LEVEL_NOTSET = 12,
/** the count of log level defined above. */
    LOG_LEVEL_COUNT = LOG_LEVEL_NOTSET + 1,
};

inline LogLevel getLevelByString(const std::string &logLevel) {
    auto levelStr = logLevel;
    std::transform(levelStr.begin(), levelStr.end(), levelStr.begin(), ::toupper);
    auto level = LOG_LEVEL_DISABLE;
    if (levelStr == "DISABLE") {
        level = LOG_LEVEL_DISABLE;
    } else if (levelStr == "FATAL") {
        level = LOG_LEVEL_FATAL;
    } else if (levelStr == "ERROR") {
        level = LOG_LEVEL_ERROR;
    } else if (levelStr == "WARN") {
        level = LOG_LEVEL_WARN;
    } else if (levelStr == "INFO") {
        level = LOG_LEVEL_INFO;
    } else if (levelStr == "DEBUG") {
        level = LOG_LEVEL_DEBUG;
    } else if (levelStr == "TRACE1") {
        level = LOG_LEVEL_TRACE1;
    } else if (levelStr == "TRACE2") {
        level = LOG_LEVEL_TRACE2;
    } else if (levelStr == "TRACE3") {
        level = LOG_LEVEL_TRACE3;
    } else if (levelStr == "SCHEDULE1") {
        level = LOG_LEVEL_SCHEDULE1;
    } else if (levelStr == "SCHEDULE2") {
        level = LOG_LEVEL_SCHEDULE2;
    } else if (levelStr == "SCHEDULE3") {
        level = LOG_LEVEL_SCHEDULE3;
    }
    return level;
}

constexpr int32_t NAVI_MAJOR_VERSION = 3;
constexpr int32_t NAVI_MINOR_VERSION = 0;
constexpr int32_t NAVI_VERSION =
    (NAVI_MAJOR_VERSION << 16) | (NAVI_MINOR_VERSION);
typedef int32_t GraphId;
constexpr GraphId PARENT_GRAPH_ID = GraphId(-1);
constexpr GraphId USER_GRAPH_ID = GraphId(-2);
constexpr GraphId RS_GRAPH_RESOURCE_GRAPH_ID = GraphId(-3);
constexpr GraphId INVALID_GRAPH_ID = GraphId(-16);

struct ScopeId {
public:
    ScopeId()
        : graphId(INVALID_GRAPH_ID)
        , scopeIndex(-1)
    {
    }
    ScopeId(GraphId graphId_, int32_t scopeIndex_)
        : graphId(graphId_)
        , scopeIndex(scopeIndex_)
    {
    }
public:
    GraphId graphId;
    int32_t scopeIndex;
};

template <typename T>
inline bool cmpxchg(volatile T *addr, T oldValue, T newValue) {
    return __sync_bool_compare_and_swap(addr, oldValue, newValue);
}

#define NAVI_MEMORY_BARRIER() __asm__ __volatile__("" ::: "memory");
#define NAVI_ALIGN(value, align)            (((value) + ((align) - 1)) & (~((align) - 1)))

extern const std::string NAVI_KERNEL_NAME;
extern const std::string NAVI_KERNEL_OUTPUT_PORT;
extern const std::string NAVI_STATE_KERNEL_NAME;
extern const std::string NAVI_STATE_KERNEL_OUTPUT_PORT;

extern const std::string NAVI_NAMESPACE_SEPERATOR;
extern const std::string NAVI_BUILDIN_BIZ;
extern const std::string NODE_PORT_SEPERATOR;
extern const std::string GROUP_PORT_SEPERATOR;
extern const std::string NODE_FINISH_PORT;
extern const std::string TERMINATOR_NODE_NAME;
extern const std::string TERMINATOR_KERNEL_NAME;
extern const std::string TERMINATOR_INPUT_PORT;
extern const std::string PLACEHOLDER_KERNEL_NAME;
extern const std::string PLACEHOLDER_OUTPUT_PORT;
extern const std::string TF_BATCH_IDENTITY_KERNEL_NAME;
extern const std::string TF_BATCH_IDENTITY_INPUT;
extern const std::string TF_BATCH_IDENTITY_OUTPUT;
extern const std::string TF_SPLIT_KERNEL;
extern const std::string IDENTITY_KERNEL;
extern const std::string IDENTITY_KERNEL_INPUT;
extern const std::string IDENTITY_KERNEL_OUTPUT;
extern const std::string NAVI_USER_GRAPH_BIZ;
extern const std::string NAVI_FORK_BIZ;
extern const std::string NAVI_FORK_NODE;
extern const std::string NAVI_SERVER_MIRROR_BIZ_PREFIX;
extern const std::string NAVI_DISCONNECTED_NODE;

extern const std::string MAP_KERNEL_INPUT_PORT;
extern const std::string MAP_KERNEL_OUTPUT_PORT;

extern const std::string NODE_CONTROL_INPUT_PORT;
extern const std::string CONTROL_DATA_TYPE;
extern const std::string CONTROL_MERGE_KERNEL;

extern const std::string BACK_PRESSURE_DATA_TYPE;

extern const std::string NODE_RESOURCE_INPUT_PORT;
extern const std::string RESOURCE_DATA_TYPE_ID;
extern const std::string RESOURCE_NOTIFY_DATA_TYPE_ID;

extern const std::string DEFAULT_SPLIT_KERNEL;
extern const std::string DEFAULT_MERGE_KERNEL;
extern const std::string PORT_KERNEL_INPUT_PORT;
extern const std::string PORT_KERNEL_OUTPUT_PORT;
extern const std::string PORT_KERNEL_PART_COUNT_KEY;

extern const std::string DEFAULT_LOG_PATTERN;
extern const std::string DEFAULT_LOG_FILE_NAME;

constexpr size_t CONTROL_PORT_FINISH_INDEX = 0;

extern const std::string NAVI_RESULT_TYPE_ID;
extern const std::string GIG_CLIENT_RESOURCE_ID;
extern const std::string GIG_SESSION_RESOURCE_ID;

extern const std::string SCOPE_TERMINATOR_NODE_NAME_PREFIX;
extern const std::string SCOPE_TERMINATOR_INPUT_PORT;

extern const std::string NAVI_RPC_METHOD_NAME;

extern const std::string NAVI_BUILDIN_ATTR_NODE;
extern const std::string NAVI_BUILDIN_ATTR_KERNEL;

extern const std::string NAVI_BUILDIN_STATIC_GRAPH_META;

extern const std::string SIMD_PADING_STR;

extern const std::string NAVI_DEBUG_THREAD_LIMIT_STR;
extern const std::string NAVI_DEBUG_MAX_INLINE_STR;
extern const std::string NAVI_DEBUG_TIMEOUT_STR;
extern const std::string NAVI_DEBUG_METRIC_STR;
extern const std::string NAVI_DEBUG_PERF_STR;
extern const std::string NAVI_DEBUG_CONTEXT_SWITCH_STR;
extern const std::string NAVI_DEBUG_PAGE_FAULT_STR;
extern const std::string NAVI_DEBUG_TRACE_STR;
extern const std::string NAVI_DEBUG_MALLOC_STR;
extern const std::string NAVI_DEBUG_SNAPSHOT_STR;
extern const std::string NAVI_DEBUG_BT_FILTER_STR;
extern const std::string NAVI_DEBUG_MALLOC_TRACE_STR;
extern const std::string NAVI_DEBUG_DELIM1_STR;
extern const std::string NAVI_DEBUG_DELIM2_STR;
extern const std::string NAVI_DEBUG_DELIM3_STR;
extern const std::string NAVI_DEBUG_PARAM_TOPO_TREE_STR;
extern const std::string NAVI_DEBUG_RESOURCE_BUILD_MODE;
extern const std::string NAVI_DEBUG_DISABLE_GIG_DEGRADE;
extern const std::string NAVI_DATA_SUMMARY_IGNORE_DATA;

typedef std::function<void()> NaviVoidFunc;
class Creator;
typedef std::function<::navi::Creator *()> NaviCreatorFunc;
typedef std::function<bool()> NaviModuleInitFunc;

class Kernel;
class Resource;
class Data;
class NaviConfigContext;
typedef NaviConfigContext KernelConfigContext;

typedef std::function<bool(void *, void *, Resource *, int64_t *, int64_t *)>
    StaticResourceBindFunc;
typedef std::function<bool(
    void *,
    void *,
    Resource *,
    int64_t *,
    int64_t *,
    void *not_used /* do not equal to other bind func */)>
    DynamicResourceBindFunc;
typedef std::function<bool(void *, void *, Data *, int64_t *)> NamedDataBindFunc;
typedef std::function<int(KernelConfigContext &ctx)> ResourceSelectFunc;
constexpr int32_t INVALID_SELECT_INDEX = 65536;
constexpr int32_t IGNORE_SELECT_INDEX = -1;
static constexpr int64_t NAVI_INVALID_CAST_OFFSET = (1LL << 63);
static constexpr int64_t NAVI_INIT_CAST_OFFSET = (1LL << 62);
inline void updateCastOffset(int64_t offset, int64_t &castOffset) {
    if (NAVI_INIT_CAST_OFFSET == castOffset) {
        castOffset = offset;
    } else if (offset != castOffset) {
        castOffset = NAVI_INVALID_CAST_OFFSET;
    }
}

struct StaticResourceBindInfo {
    std::string name;
    bool require = true;
    StaticResourceBindFunc binder;
};

struct DynamicResourceBindInfo {
    std::string group;
    DynamicResourceBindFunc binder;
};

struct NamedDataBindInfo {
    std::string name;
    bool require = true;
    NamedDataBindFunc binder;
};

struct SelectResourceBindInfo {
    std::vector<std::string> candidates;
    bool require = true;
    ResourceSelectFunc selector;
    StaticResourceBindFunc binder;
};

struct BindInfos {
    std::vector<StaticResourceBindInfo> staticBindVec;
    std::vector<DynamicResourceBindInfo> dynamicBindVec;
    std::vector<NamedDataBindInfo> namedDataBindVec;
    std::vector<SelectResourceBindInfo> selectBindVec;
};

constexpr uint32_t DEFAULT_THREAD_LIMIT = 100000u;
constexpr size_t DEFAULT_THREAD_NUMBER = 0u; // sysconf(_SC_NPROCESSORS_ONLN)
constexpr size_t DEFAULT_MIN_THREAD_NUMBER = 20u;
constexpr size_t DEFAULT_MAX_THREAD_NUMBER = 104u;
constexpr size_t DEFAULT_QUEUE_SIZE = 1000u;
constexpr size_t DEFAULT_PROCESSING_SIZE = 300u;
constexpr int32_t DEFAULT_THREAD_NICE_VALUE = 0;
constexpr int64_t FACTOR_MS_TO_US = 1000;
constexpr int64_t DEFAULT_TIMEOUT_MS = 10000000000000;
constexpr int32_t DEFAULT_MAX_INLINE = 3;
constexpr float RPC_TIMEOUT_DECAY_FACTOR = 0.9f;
constexpr int64_t SUBGRAPH_TIMEOUT_DECAY_MS = 30;
constexpr int64_t DESTRUCT_DELAY_US = 200 * 1000;
constexpr int64_t DEFAULT_HISTORY_MAX_UPDATE_MEM_SIZE = 64 * 1024 * 1024;
constexpr int64_t DEFAULT_HISTORY_MAX_UPDATE_EVENT_COUNT = 1024;
constexpr int64_t DEFAULT_HISTORY_MAX_QUERY_MEM_SIZE = 32 * 1024 * 1024;
constexpr int64_t DEFAULT_HISTORY_MAX_QUERY_EVENT_COUNT = 2048;
constexpr int64_t DEFAULT_HISTORY_QUERY_SAMPLE_INTERVAL_US = 1000 * 1000;

enum RegistryType : int32_t {
    RT_RESOURCE = 0,
    RT_KERNEL = 1,
    RT_TYPE = 2,
    RT_MAX = 3,
};

template <int N>
class DependMatcher {
public:
    unsigned int value() { return N; }
};
}

#define NAVI_KERNEL_ERROR_RETURN(code, level, format, args...)                 \
    {                                                                          \
        NAVI_KERNEL_LOG(level, format, ##args);                                \
        return code;                                                           \
    }

namespace google {
namespace protobuf
{
class Arena;
}
}

typedef std::shared_ptr<google::protobuf::Arena> ArenaPtr;

enum TestMode : int32_t {
    TM_NONE = 0,
    TM_KERNEL_TEST = 1,
    TM_RESOURCE_TEST = 2,
    TM_UNKNOWN = 10
};
