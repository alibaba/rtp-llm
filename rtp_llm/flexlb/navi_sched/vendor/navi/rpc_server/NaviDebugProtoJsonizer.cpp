#include "navi/rpc_server/NaviDebugProtoJsonizer.h"
#include "navi/proto/GraphVis.pb.h"

namespace navi {

NaviDebugProtoJsonizer::NaviDebugProtoJsonizer(
    const std::shared_ptr<GraphVisDef> &visDef,
    const http_arpc::ProtoJsonizerPtr &realProtoJsonizer)
    : _visDef(visDef)
    , _realProtoJsonizer(realProtoJsonizer)
{
}

NaviDebugProtoJsonizer::~NaviDebugProtoJsonizer() {
}

bool NaviDebugProtoJsonizer::fromJson(const std::string &jsonStr,
                                      google::protobuf::Message *message)
{
    assert(false);
    return false;
}

std::string NaviDebugProtoJsonizer::toJson(const google::protobuf::Message &message) {
    auto response = _realProtoJsonizer->toJson(message);
    if (1 != _visDef->nodes_size()) {
        return response;
    }
    auto nodeDef = _visDef->mutable_nodes(0);
    nodeDef->set_user_result(response);
    return _visDef->SerializeAsString();
}

}

