#include "navi/ops/DefaultMergeKernel.h"
#include "navi/engine/Port.h"
#include "navi/engine/NaviRegisterMacro.h"
#include "navi/util/ReadyBitMap.h"
#include "navi/util/CommonUtil.h"

namespace navi {

DefaultMergeKernel::DefaultMergeKernel()
        : _loopIndex(0)
{
}

DefaultMergeKernel::~DefaultMergeKernel() {
    InlineReadyBitMap::freeReadyBitMap(_dataReadyMap);
}

std::string DefaultMergeKernel::getName() const {
    return DEFAULT_MERGE_KERNEL;
}

std::string DefaultMergeKernel::dataType() const {
    return "";
}

InputTypeDef DefaultMergeKernel::inputType() const {
    return IT_OPTIONAL;
}

ErrorCode DefaultMergeKernel::doInit(KernelInitContext &ctx) {
    if (_port) {
        _dataReadyMap = _port->getPartInfo().createReadyBitMap();
    } else {
        _dataReadyMap = InlineReadyBitMap::createReadyBitMap(getUsedPartCount());
    }
    _dataReadyMap.setFinish(false);
    return EC_NONE;
}

ErrorCode DefaultMergeKernel::compute(KernelComputeContext &ctx) {
    auto usedPartCount = getUsedPartCount();
    for (auto loopCount = 0; loopCount < usedPartCount; ++loopCount) {
        StreamData streamData;
        auto index = getUsedPartId(_loopIndex);
        auto ec = getData(ctx, index, streamData);
        _loopIndex++;
        if (_loopIndex >= usedPartCount) {
            _loopIndex = 0;
        }
        if (EC_NONE != ec) {
            NAVI_KERNEL_LOG(ERROR, "get data failed, index: %d", index);
            return ec;
        }
        if (!streamData.hasValue) {
            continue;
        }
        if (streamData.eof) {
            _dataReadyMap.setFinish(index, true);
        }
        auto eof = _dataReadyMap.isFinish();
        NAVI_KERNEL_LOG(SCHEDULE1,
                        "input data partId: %d, loopIndex: %d, output data [%p] eof [%d]",
                        index,
                        _loopIndex,
                        streamData.data.get(),
                        eof);
        if (streamData.data || eof) {
            if (!ctx.setOutput(OUTPUT_PORT, streamData.data, eof)) {
                NAVI_KERNEL_LOG(ERROR, "setOutput failed");
                return EC_UNKNOWN;
            }
        }
        return EC_NONE;
    }
    ctx.setIgnoreDeadlock();
    return EC_NONE;
}

ErrorCode DefaultMergeKernel::doCompute(const std::vector<StreamData> &dataVec,
                                        StreamData &outputData)
{
    assert(false);
    return EC_NONE;
}

REGISTER_BUILDIN_KERNEL(DefaultMergeKernel);

}
