#include "navi/engine/PortMergeKernel.h"
#include "navi/engine/LocalPortBase.h"
#include "navi/engine/Node.h"
#include "navi/engine/LocalSubGraph.h"
#include "navi/engine/SubGraphBase.h"
#include "navi/engine/Graph.h"
#include "navi/engine/PartInfo.h"

namespace navi {

IndexType PortMergeKernel::INPUT_PORT_GROUP = 0;
IndexType PortMergeKernel::OUTPUT_PORT = 0;

PortMergeKernel::PortMergeKernel()
        : _port(nullptr)
{
}

PortMergeKernel::~PortMergeKernel() {
}

void PortMergeKernel::def(KernelDefBuilder &builder) const {
    const auto &type = dataType();
    builder.name(getName())
        .inputGroup(PORT_KERNEL_INPUT_PORT, type, inputType())
        .output(PORT_KERNEL_OUTPUT_PORT, type);
    auto infoVec = dependResources();
    for (const auto &info : infoVec) {
        builder.dependOn(info.name, info.require, info.binder);
    }
}

std::vector<StaticResourceBindInfo> PortMergeKernel::dependResources() const {
    return {};
}

InputTypeDef PortMergeKernel::inputType() const {
    return IT_REQUIRE;
}

bool PortMergeKernel::config(KernelConfigContext &ctx) {
    return doConfig(ctx);
}

bool PortMergeKernel::doConfig(KernelConfigContext &ctx) {
    return true;
}

ErrorCode PortMergeKernel::init(KernelInitContext &ctx) {
    _port = ctx._node->getPort();
    if (!_port) {
        if (!ctx.getInputGroupSize(0, _inputGroupSize)) {
            NAVI_KERNEL_LOG(ERROR, "get input group [0] size failed");
            return EC_UNKNOWN;
        }
    }
    return doInit(ctx);
}

ErrorCode PortMergeKernel::doInit(KernelInitContext &ctx) {
    _dataVec.resize(getUsedPartCount());
    return EC_NONE;
}

ErrorCode PortMergeKernel::compute(KernelComputeContext &ctx) {
    auto ec = fillInputs(ctx);
    if (ec != EC_NONE) {
        return ec;
    }
    StreamData streamData;
    ec = doCompute(_dataVec, streamData);
    for (auto &streamData : _dataVec) {
        streamData.reset();
    }
    if (EC_NONE != ec) {
        return ec;
    }
    if (!ctx.setOutput(OUTPUT_PORT, streamData.data, streamData.eof)) {
        NAVI_KERNEL_LOG(ERROR, "setOutput failed");
        return EC_UNKNOWN;
    }
    return EC_NONE;
}

ErrorCode PortMergeKernel::fillInputs(KernelComputeContext &ctx) {
    auto usedPartCount = getUsedPartCount();
    for (auto index = 0; index < usedPartCount; ++index) {
        auto partId = getUsedPartId(index);
        auto &streamData = _dataVec[index];
        bool oldEof = streamData.eof;
        auto ec = getData(ctx, partId, streamData);
        if (EC_NONE != ec) {
            return ec;
        }
        if (oldEof && !streamData.eof) {
            if (_port) {
                NAVI_KERNEL_LOG(ERROR,
                                "eof blink, partId: %d, portId: %d, port key: %s",
                                partId,
                                _port->getPortId(),
                                _port->getPortKey().c_str());
            } else {
                NAVI_KERNEL_LOG(ERROR, "eof blink, index: %d", partId);
            }
            return EC_UNKNOWN;
        }
    }
    return EC_NONE;
}

ErrorCode PortMergeKernel::getData(KernelComputeContext &ctx,
                                   NaviPartId partId,
                                   StreamData &streamData)
{
    if (_port) {
        ctx._deadLock = false;
        auto ec = _port->getData(partId, streamData.data, streamData.eof);
        if (EC_NO_DATA == ec) {
            return EC_NONE;
        } else if (EC_NONE != ec) {
            NAVI_KERNEL_LOG(ERROR, "get port data failed ec [%s], partId: %d",
                            CommonUtil::getErrorString(ec), partId);
            return ec;
        }
        streamData.hasValue = true;
    } else {
        PortIndex index(partId, 0);
        if (!ctx.getInput(index, streamData)) {
            NAVI_KERNEL_LOG(ERROR, "getInput failed, index: %d", partId);
            return EC_UNKNOWN;
        }
    }
    return EC_NONE;
}

NaviPartId PortMergeKernel::getUsedPartCount() const {
    if (_port) {
        return _port->getPartInfo().getUsedPartCount();
    } else {
        return _inputGroupSize;
    }
}

NaviPartId PortMergeKernel::getUsedPartId(NaviPartId index) const {
    if (_port) {
        return _port->getPartInfo().getUsedPartId(index);
    } else {
        return index;
    }
}

}
