package org.flexlb.cache.hash;

import org.flexlb.cache.domain.BlockHashConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;

import java.util.List;

/** Resolves the primary block-cache keys used to schedule a request. */
@Component
public class RequestBlockHashService {

    private final BlockHashConfigResolver blockHashConfigResolver;
    private final BlockHashExecutor blockHashExecutor;

    public RequestBlockHashService(
            BlockHashConfigResolver blockHashConfigResolver,
            BlockHashExecutor blockHashExecutor) {
        this.blockHashConfigResolver = blockHashConfigResolver;
        this.blockHashExecutor = blockHashExecutor;
    }

    public Mono<Void> prepareBlockCacheKeys(BalanceContext context) {
        return Mono.defer(() -> prepareRequest(context));
    }

    private Mono<Void> prepareRequest(BalanceContext context) {
        if (context == null) {
            return Mono.error(new IllegalArgumentException("context must not be null"));
        }
        Request request = context.getRequest();
        if (request == null) {
            return Mono.error(new IllegalArgumentException("request must not be null"));
        }
        List<Long> blockCacheKeys = request.getBlockCacheKeys();
        if (blockCacheKeys != null && !blockCacheKeys.isEmpty()) {
            return prepareProvidedBlockCacheKeys(request);
        }

        int[] inputIds = request.getInputIds();
        if (inputIds == null || inputIds.length == 0) {
            return Mono.error(new IllegalArgumentException(
                    "block_cache_keys and input_ids must not both be empty"));
        }

        BlockHashConfig hashConfig = blockHashConfigResolver.resolve();
        long blockSize = request.getBlockSize() > 0
                ? request.getBlockSize()
                : hashConfig.blockSize();
        request.setBlockSize(blockSize);
        return blockHashExecutor
                .calculate(inputIds, blockSize, hashConfig.lookaheadTokens())
                .doOnNext(result -> {
                    request.setBlockCacheKeys(result.blockCacheKeys());
                    request.setInputIds(null);
                })
                .then();
    }

    private Mono<Void> prepareProvidedBlockCacheKeys(Request request) {
        if (request.getBlockSize() <= 0) {
            return Mono.error(new IllegalArgumentException(
                    "block_size must be greater than 0 when block_cache_keys are provided"));
        }
        request.setInputIds(null);
        return Mono.empty();
    }
}
