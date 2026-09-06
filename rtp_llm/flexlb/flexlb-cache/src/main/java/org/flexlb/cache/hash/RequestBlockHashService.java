package org.flexlb.cache.hash;

import org.flexlb.cache.domain.BlockHashConfig;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.config.LocalStandbyConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;

import java.util.List;

/** Resolves the block cache keys used to schedule a request. */
@Component
public class RequestBlockHashService {

    private final BlockHashConfigResolver blockHashConfigResolver;
    private final BlockHashExecutor blockHashExecutor;
    private final LocalStandbyHashService localStandbyHashService;
    private final boolean localStandbyEnabled;
    private final long configuredLocalStandbyBlockSize;

    public RequestBlockHashService(
            BlockHashConfigResolver blockHashConfigResolver,
            BlockHashExecutor blockHashExecutor,
            LocalStandbyHashService localStandbyHashService,
            CacheMatchConfiguration configuration) {
        this.blockHashConfigResolver = blockHashConfigResolver;
        this.blockHashExecutor = blockHashExecutor;
        this.localStandbyHashService = localStandbyHashService;
        LocalStandbyConfig localStandbyConfig = configuration.getLocalStandbyConfig();
        this.localStandbyEnabled = configuration.isLocalStandbyEnabled();
        this.configuredLocalStandbyBlockSize = localStandbyConfig == null
                ? 0
                : localStandbyConfig.getBlockSize();
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
            return prepareProvidedBlockCacheKeys(request, blockCacheKeys);
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

        long localStandbyBlockSize = configuredLocalStandbyBlockSize > 0
                ? configuredLocalStandbyBlockSize
                : blockSize;
        boolean reusePrimaryHash = localStandbyEnabled
                && localStandbyBlockSize == blockSize;
        if (localStandbyEnabled) {
            request.setLocalStandbyBlockSize(localStandbyBlockSize);
            if (!reusePrimaryHash) {
                localStandbyHashService.submit(
                        request,
                        inputIds,
                        localStandbyBlockSize,
                        hashConfig.lookaheadTokens());
            }
        }

        return blockHashExecutor
                .calculate(inputIds, blockSize, hashConfig.lookaheadTokens())
                .doOnNext(result -> {
                    request.setBlockCacheKeys(result.blockCacheKeys());
                    if (reusePrimaryHash) {
                        request.setLocalStandbyBlockCacheKeys(result.blockCacheKeys());
                        request.setLocalStandbyCacheableBlockCacheKeys(
                                blockHashExecutor.cacheablePrefix(
                                        result.blockCacheKeys(),
                                        inputIds.length,
                                        blockSize,
                                        hashConfig.lookaheadTokens()));
                    }
                    request.setInputIds(null);
                    context.recordBlockHashTiming(
                            result.queueWaitTimeUs(), result.executionTimeUs());
                })
                .then();
    }

    private Mono<Void> prepareProvidedBlockCacheKeys(
            Request request, List<Long> blockCacheKeys) {
        long requestBlockSize = request.getBlockSize();
        if (requestBlockSize <= 0) {
            return Mono.error(new IllegalArgumentException(
                    "block_size must be greater than 0 when block_cache_keys are provided"));
        }
        if (localStandbyEnabled) {
            request.setLocalStandbyBlockSize(requestBlockSize);
            request.setLocalStandbyBlockCacheKeys(blockCacheKeys);
            request.setLocalStandbyCacheableBlockCacheKeys(blockCacheKeys);
        }
        request.setInputIds(null);
        return Mono.empty();
    }
}
