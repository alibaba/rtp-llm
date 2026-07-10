package org.flexlb.httpserver;

import org.flexlb.dao.loadbalance.Request;
import org.flexlb.util.BlockCacheKeyCalculator;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * Resolves schedule request cache keys before local routing.
 */
@Component
public class ScheduleRequestPreprocessor {

    private final WorkerBlockSizeResolver blockSizeResolver;

    public ScheduleRequestPreprocessor(WorkerBlockSizeResolver blockSizeResolver) {
        this.blockSizeResolver = blockSizeResolver;
    }

    public void prepare(Request request) {
        if (request == null) {
            throw new IllegalArgumentException("request must not be null");
        }

        List<Long> blockCacheKeys = request.getBlockCacheKeys();
        if (blockCacheKeys != null && !blockCacheKeys.isEmpty()) {
            request.setInputIds(null);
            return;
        }

        List<Long> inputIds = request.getInputIds();
        if (inputIds == null || inputIds.isEmpty()) {
            throw new IllegalArgumentException(
                    "block_cache_keys and input_ids must not both be empty");
        }

        long blockSize = request.getBlockSize();
        if (blockSize <= 0) {
            blockSize = blockSizeResolver.resolve();
        }

        request.setBlockCacheKeys(BlockCacheKeyCalculator.calculate(inputIds, blockSize));
        request.setInputIds(null);
    }
}
