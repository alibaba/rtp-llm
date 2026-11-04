#include "navi/ops/ResourceKernelDef.h"

namespace navi {

const std::string RESOURCE_CREATE_KERNEL = "navi.r_create";
const std::string RESOURCE_CREATE_OUTPUT = "out";
const std::string RESOURCE_PRODUCE_OUTPUT = "produce_out";
const std::string RESOURCE_ATTR_NAME = "resource_name";
const std::string RESOURCE_ATTR_REQUIRE = "require";
const std::string RESOURCE_ATTR_PRODUCE_NAME = "produce_name";
const std::string RESOURCE_ATTR_PRODUCE_REQUIRE = "produce_require";
const std::string RESOURCE_ATTR_INIT = "init";
const std::string RESOURCE_ATTR_SUBSCRIBE = "subscribe";
const std::string RESOURCE_ATTR_REQUIRE_NODE = "require_node";
const std::string RESOURCE_ATTR_REQUIRE_NODE_INDEX = "require_node_index";

const std::string RESOURCE_SAVE_KERNEL = "navi.r_save";
const std::string RESOURCE_SAVE_KERNEL_INPUT = "in";
const std::string RESOURCE_SAVE_KERNEL_OUTPUT = "out";

const std::string RESOURCE_PUBLISH_KERNEL = "navi.r_publish";
const std::string RESOURCE_PUBLISH_KERNEL_INPUT = "in";
const std::string RESOURCE_PUBLISH_KERNEL_OUTPUT = "out";

const std::string RESOURCE_FLUSH_KERNEL = "navi.r_flush";
const std::string RESOURCE_FLUSH_KERNEL_OUTPUT = "out";

}
