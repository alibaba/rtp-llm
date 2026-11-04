#include "navi/config/GraphConfig.h"

namespace navi {

GraphConfig::GraphConfig() {
}

GraphConfig::~GraphConfig() {
}

void GraphConfig::Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) {
    json.Jsonize("graph_file", graphFile, graphFile);
}

}

