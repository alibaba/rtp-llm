#include "navi/engine/LocalSubGraphInternal.h"
#include "navi/engine/Node.h"

namespace navi {

LocalSubGraphInternal::LocalSubGraphInternal()
    : subGraphDef(nullptr)
    , graphStarted(false)
{}

LocalSubGraphInternal::~LocalSubGraphInternal() {
    for (const auto &node : nodeVec) {
        delete node;
    }
    for (const auto &pair : edges) {
        delete pair.second;
    }
    delete subGraphDef;
}

NodeInternal *LocalSubGraphInternal::getNode(const std::string &nodeName,
                                             int32_t portNodeId) const {
    if (portNodeId <= 0) {
        auto it = nodeMap.find(nodeName);
        if (it == nodeMap.end()) {
            return nullptr;
        }
        return it->second;
    } else {
        auto index = portNodeId - 1;
        if (index >= nodeVec.size()) {
            return nullptr;
        } else {
            return nodeVec[index];
        }
    }
}

bool LocalSubGraphInternal::postInit(CreatorManager *creatorManager) {
    for (const auto &node : nodeVec) {
        if (!node->postInit()) {
            return false;
        }
    }
    if (!checkEdge(creatorManager)) {
        return false;
    }
    initSize();
    return true;
}

bool LocalSubGraphInternal::checkEdge(CreatorManager *creatorManager) const {
    bool ret = true;
    for (const auto &pair : edges) {
        ret = pair.second->checkType(creatorManager) && ret;
    }
    return ret;
}

void LocalSubGraphInternal::initSize() {
    size_t totalSize = 0;
    if (subGraphDef) {
        totalSize += subGraphDef->ByteSizeLong();
    }
    totalSize += sizeof(NodeInternal) * nodeMap.size();
    totalSize += sizeof(decltype(nodeMap)::value_type) * nodeMap.size();
    totalSize += sizeof(decltype(nodeVec)::value_type) * nodeVec.capacity();
    totalSize += sizeof(decltype(edgeVec)::value_type) * edgeVec.capacity();
    totalSize += sizeof(decltype(edges)::value_type) * edges.size();
    totalSize += sizeof(decltype(edgeInternalVec)::value_type) * edgeInternalVec.capacity();
    totalSize += sizeof(decltype(scopeInfoMap)::value_type) * scopeInfoMap.size();
    totalSize += sizeof(decltype(startedNodes)::value_type) * startedNodes.capacity();
    totalSize += sizeof(LocalSubGraphInternal);
    size = totalSize;
}

void LocalSubGraphInternal::fillProto(GraphCacheSummaryDef &summaryDef) const {
    summaryDef.set_hash_key(std::to_string(hashKey));
    summaryDef.set_size(getSize());
    summaryDef.set_node_count(nodeVec.size());
    summaryDef.set_edge_count(edgeInternalVec.size());
    summaryDef.set_connect_count(edgeVec.size());
    summaryDef.set_last_access_time(lastAccessTime);
    NaviError::fillSessionId(summaryDef.mutable_session(), session);
    summaryDef.set_peer_ip(peerIp);
    summaryDef.set_malloc_size(mallocSize);
}

}

