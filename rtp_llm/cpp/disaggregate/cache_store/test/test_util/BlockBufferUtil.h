#include <vector>

#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBuffer.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "rtp_llm/cpp/disaggregate/cache_store/test/test_util/DeviceUtil.h"

namespace rtp_llm {

// BlockBufferUtil use for test, mock real alloc memory
class BlockBufferUtil {
public:
    BlockBufferUtil(const std::shared_ptr<MemoryUtil>& memory_util, const std::shared_ptr<DeviceUtil>& device_util);
    ~BlockBufferUtil();

public:
    std::shared_ptr<BlockBuffer> makeBlockBuffer(const std::string& key, uint32_t len, char val, bool gpu);

    std::vector<std::shared_ptr<RequestBlockBuffer>>
    makeRequestBlockBufferVec(const std::string& requestid, int layer_num, int block_num, uint32_t block_size);

    bool verifyBlock(
        const std::shared_ptr<BlockBuffer>& block, const std::string& key, uint32_t len, bool gpu_mem, char val);

private:
    std::shared_ptr<MemoryUtil> memory_util_;
    std::shared_ptr<DeviceUtil> device_util_;
};

}  // namespace rtp_llm