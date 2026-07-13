package org.flexlb.httpserver;

import org.flexlb.dao.loadbalance.Request;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;

import java.util.List;

/**
 * Resolves schedule request cache keys before local routing.
 */
@Component
public class ScheduleRequestPreprocessor {

    private final WorkerBlockSizeResolver blockSizeResolver;
    private final BlockHashExecutor blockHashExecutor;

    public ScheduleRequestPreprocessor(
            WorkerBlockSizeResolver blockSizeResolver,
            BlockHashExecutor blockHashExecutor) {
        this.blockSizeResolver = blockSizeResolver;
        this.blockHashExecutor = blockHashExecutor;
    }

    public Mono<Void> prepare(Request request) {
        return Mono.defer(() -> prepareRequest(request));
    }

    private Mono<Void> prepareRequest(Request request) {
        if (request == null) {
            return Mono.error(new IllegalArgumentException("request must not be null"));
        }
        List<Long> blockCacheKeys = request.getBlockCacheKeys();
        if (blockCacheKeys != null && !blockCacheKeys.isEmpty()) {
            request.setInputIds(null);
            return Mono.empty();
        }

        List<Long> inputIds = request.getInputIds();
        if (inputIds == null || inputIds.isEmpty()) {
            return Mono.error(new IllegalArgumentException(
                    "block_cache_keys and input_ids must not both be empty"));
        }

        long blockSize = request.getBlockSize();
        if (blockSize <= 0) {
            blockSize = blockSizeResolver.resolve();
        }

        return blockHashExecutor.calculate(inputIds, blockSize)
                .doOnNext(calculatedKeys -> {
                    request.setBlockCacheKeys(calculatedKeys);
                    request.setInputIds(null);
                })
                .then();
    }
}
