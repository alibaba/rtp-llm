#include "navi/common.h"

namespace navi {

std::ostream &operator<<(std::ostream &os, const PortIndex &portIndex) {
    return os << (int)portIndex.isGroup() << "," << (int)portIndex.isControl() << "," << portIndex.group() << ","
              << portIndex.index();
}

const std::string NAVI_KERNEL_NAME = "navi.NaviKernel";
const std::string NAVI_KERNEL_OUTPUT_PORT = "navi.NaviKernel_out";
const std::string NAVI_STATE_KERNEL_NAME = "navi.NaviStateKernel";
const std::string NAVI_STATE_KERNEL_OUTPUT_PORT = "navi.NaviStateKernel_out";

const std::string NAVI_NAMESPACE_SEPERATOR = ".";
const std::string NAVI_BUILDIN_BIZ = "navi.buildin_biz";
const std::string NODE_PORT_SEPERATOR = ".";
const std::string GROUP_PORT_SEPERATOR = ":";
const std::string NODE_FINISH_PORT = "node@finish";
const std::string TERMINATOR_NODE_NAME = "navi.terminator";
const std::string TERMINATOR_KERNEL_NAME = "navi.TerminatorKernel";
const std::string TERMINATOR_INPUT_PORT = "navi.terminator@input";
const std::string PLACEHOLDER_KERNEL_NAME = "navi.placeholder";
const std::string PLACEHOLDER_OUTPUT_PORT = "navi.placeholder_out";
const std::string TF_BATCH_IDENTITY_KERNEL_NAME = "navi.TFBatchIdentityKernel";
const std::string TF_BATCH_IDENTITY_INPUT = "input";
const std::string TF_BATCH_IDENTITY_OUTPUT = "output";
const std::string TF_SPLIT_KERNEL = "navi.TFSplitKernel";
const std::string IDENTITY_KERNEL = "navi.IdentityKernel";
const std::string IDENTITY_KERNEL_INPUT = "input0";
const std::string IDENTITY_KERNEL_OUTPUT = "output0";
const std::string NAVI_USER_GRAPH_BIZ = "navi.user_biz";
const std::string NAVI_FORK_BIZ = "navi.fork";
const std::string NAVI_FORK_NODE = "navi.fork_node";
const std::string NAVI_SERVER_MIRROR_BIZ_PREFIX = "mirror_biz@";
const std::string NAVI_DISCONNECTED_NODE = "navi.disconnected";

const std::string MAP_KERNEL_INPUT_PORT = "in";
const std::string MAP_KERNEL_OUTPUT_PORT = "out";

const std::string NODE_CONTROL_INPUT_PORT = "node@control_input";
const std::string CONTROL_DATA_TYPE = "node@control_data_type";
const std::string CONTROL_MERGE_KERNEL = "navi.control_merge";

const std::string BACK_PRESSURE_DATA_TYPE = "node@back_pressure_data_type";

const std::string NODE_RESOURCE_INPUT_PORT = "node@resource_input";
const std::string RESOURCE_DATA_TYPE_ID = "navi.resource_data_t";
const std::string RESOURCE_NOTIFY_DATA_TYPE_ID = "navi.resource_notify_data_t";

const std::string DEFAULT_SPLIT_KERNEL = "navi.split";
const std::string DEFAULT_MERGE_KERNEL = "navi.merge";
const std::string PORT_KERNEL_INPUT_PORT = "in";
const std::string PORT_KERNEL_OUTPUT_PORT = "out";
const std::string PORT_KERNEL_PART_COUNT_KEY = "part_count";

const std::string DEFAULT_LOG_PATTERN = "[%%c] [%%d] [%%l] [%%t,%%F -- %%f(%%o):%%n] [%%z] [%%m] [%%b]";
const std::string DEFAULT_LOG_FILE_NAME = "logs/navi.log";

const std::string NAVI_RESULT_TYPE_ID = "navi.buildin.type.navi_result";
const std::string GIG_CLIENT_RESOURCE_ID = "navi.buildin.gig_client";
const std::string GIG_SESSION_RESOURCE_ID = "navi.buildin.gig_session";

const std::string SCOPE_TERMINATOR_NODE_NAME_PREFIX = "navi.scope_terminator:";
const std::string SCOPE_TERMINATOR_INPUT_PORT = "navi.in";

const std::string NAVI_RPC_METHOD_NAME = "navi";

const std::string NAVI_BUILDIN_ATTR_NODE = "navi.buildin.attr.node";
const std::string NAVI_BUILDIN_ATTR_KERNEL = "navi.buildin.attr.kernel";

const std::string NAVI_BUILDIN_STATIC_GRAPH_META = "navi.buildin.static_graph.meta";

const std::string SIMD_PADING_STR(128, '\0');

const std::string NAVI_DEBUG_THREAD_LIMIT_STR = "navi_thread_limit";
const std::string NAVI_DEBUG_MAX_INLINE_STR = "navi_max_inline";
const std::string NAVI_DEBUG_TIMEOUT_STR = "navi_timeout";
const std::string NAVI_DEBUG_METRIC_STR = "navi_metric";
const std::string NAVI_DEBUG_PERF_STR = "navi_perf";
const std::string NAVI_DEBUG_CONTEXT_SWITCH_STR = "navi_context_switch";
const std::string NAVI_DEBUG_PAGE_FAULT_STR = "navi_page_fault";
const std::string NAVI_DEBUG_TRACE_STR = "navi_trace";
const std::string NAVI_DEBUG_MALLOC_STR = "navi_malloc";
const std::string NAVI_DEBUG_SNAPSHOT_STR = "navi_snapshot";
const std::string NAVI_DEBUG_DELIM1_STR = ",";
const std::string NAVI_DEBUG_DELIM2_STR = ":";
const std::string NAVI_DEBUG_DELIM3_STR = "@";
const std::string NAVI_DEBUG_BT_FILTER_STR = "navi_bt_filter";
const std::string NAVI_DEBUG_MALLOC_TRACE_STR = "navi_malloc_trace";
const std::string NAVI_DEBUG_PARAM_TOPO_TREE_STR = "navi_param_topo_tree";
const std::string NAVI_DEBUG_RESOURCE_BUILD_MODE = "navi_resource_build_mode";
const std::string NAVI_DEBUG_DISABLE_GIG_DEGRADE = "navi_disable_gig_degrade";
const std::string NAVI_DATA_SUMMARY_IGNORE_DATA = "navi_ignore_data";

}
