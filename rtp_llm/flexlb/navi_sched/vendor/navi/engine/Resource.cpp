#include "navi/log/NaviLogger.h"
#include "navi/engine/Resource.h"
#include "navi/engine/ResourceMap.h"

namespace navi {

void Resource::initName() {
    if (!_name.empty()) {
        return;
    }
    navi::ResourceDef resourceDef;
    ResourceDefBuilder builder(&resourceDef, true);
    def(builder);
    setName(resourceDef.name());
}

void Resource::initLogger(NaviObjectLogger &logger, const std::string &prefix) {
    if (NAVI_TLS_LOGGER) {
        logger.object = this;
        logger.prefix = prefix;
        logger.logger = NAVI_TLS_LOGGER->logger;
    }
}

ErrorCode ExternalResource::init(ResourceInitContext &ctx) {
    NAVI_KERNEL_LOG(ERROR, "root resource do not support create by resource manager");
    return EC_ABORT;
}

ExternalResource::ExternalResource(const std::string &name, ResourceStage stage)
    : _stage(stage)
{
    setName(name);
}

void ExternalResource::def(ResourceDefBuilder &builder) const {
    builder.name(_name, _stage);
}

}
