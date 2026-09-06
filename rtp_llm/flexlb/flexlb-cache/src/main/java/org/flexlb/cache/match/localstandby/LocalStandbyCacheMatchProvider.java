package org.flexlb.cache.match.localstandby;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.cache.domain.CacheMatchQuery;
import org.flexlb.cache.domain.CacheMatchResult;
import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.cache.domain.LocalStandbyHashResult;
import org.flexlb.cache.hash.LocalStandbyHashService;
import org.flexlb.cache.match.CacheMatchProvider;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.dao.cache.HostCacheMatch;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.route.LocalStandbyConfig;
import org.flexlb.dao.route.RoleType;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * Cache matching backed by approximate request-derived local standby metadata.
 */
@Slf4j
@Component
public class LocalStandbyCacheMatchProvider implements CacheMatchProvider {

    private final boolean enabled;
    private final LocalStandbyCacheManager cacheManager;
    private final LocalStandbyHashService localStandbyHashService;
    private final ThreadPoolExecutor asyncMatchExecutor;
    private final ThreadPoolExecutor updateExecutor;

    public LocalStandbyCacheMatchProvider(
            CacheMatchConfiguration configuration,
            LocalStandbyCacheManager cacheManager,
            LocalStandbyHashService localStandbyHashService) {
        LocalStandbyConfig config = configuration.getLocalStandbyConfig();
        this.enabled = configuration.isLocalStandbyEnabled();
        this.cacheManager = cacheManager;
        this.localStandbyHashService = localStandbyHashService;
        int queueCapacity = enabled
                ? config.getAsyncQueueCapacity()
                : LocalStandbyConfig.DEFAULT_ASYNC_QUEUE_CAPACITY;
        this.asyncMatchExecutor = createExecutor(queueCapacity, "local-standby-cache-matcher");
        this.updateExecutor = createExecutor(queueCapacity, "local-standby-cache-updater");
    }

    @Override
    public CacheMatchSource source() {
        return CacheMatchSource.LOCAL_STANDBY;
    }

    @Override
    public Map<String, HostCacheMatch> findMatchingEngines(String requestId, List<Long> blockCacheKeys,
                                                           long blockSize, RoleType roleType, String group) {
        return HostCacheMatch.fromLocalMatches(cacheManager.findMatchingEngines(blockCacheKeys, roleType, group));
    }

    public CompletableFuture<CacheMatchResult> asyncLocalStandbyMatch(CacheMatchQuery query) {
        if (!enabled || query == null || query.localStandbyBlockSize() <= 0) {
            return CompletableFuture.completedFuture(CacheMatchResult.empty(CacheMatchSource.LOCAL_STANDBY));
        }

        long startTimeNs = System.nanoTime();
        try {
            return localStandbyHashService
                    .getHashResult(query.requestId(), query.localStandbyBlockCacheKeys(), query.localStandbyBlockSize())
                    .thenApplyAsync(hashResult -> {
                        if (hashResult.blockCacheKeys().isEmpty() || hashResult.blockSize() <= 0) {
                            long queryTimeUs = (System.nanoTime() - startTimeNs) / 1_000;
                            return CacheMatchResult.failed(CacheMatchSource.LOCAL_STANDBY, queryTimeUs);
                        }

                        Map<String, HostCacheMatch> matches = findMatchingEngines(
                                query.requestId(),
                                hashResult.blockCacheKeys(),
                                hashResult.blockSize(),
                                query.roleType(),
                                query.group());
                        long queryTimeUs = (System.nanoTime() - startTimeNs) / 1_000;
                        return new CacheMatchResult(
                                matches, CacheMatchSource.LOCAL_STANDBY, queryTimeUs, hashResult.blockSize());
                    }, asyncMatchExecutor
                    );
        } catch (RejectedExecutionException e) {
            log.warn("Local Standby match queue is full, requestId={}", query.requestId());
            long queryTimeUs = (System.nanoTime() - startTimeNs) / 1_000;
            return CompletableFuture.completedFuture(CacheMatchResult.failed(CacheMatchSource.LOCAL_STANDBY, queryTimeUs));
        }
    }

    public void updateFromRoutedRequest(Request request, List<ServerStatus> selectedWorkers) {
        try {
            localStandbyHashService.getHashResult(String.valueOf(request.getRequestId()),
                            request.getLocalStandbyBlockCacheKeys(),
                            request.getLocalStandbyBlockSize())
                    .thenAcceptAsync(
                            hashResult -> updateCacheMetadataNow(request, hashResult, selectedWorkers),
                            updateExecutor)
                    .exceptionally(error -> {
                        log.warn("Failed to update Local Standby cache metadata, requestId={}", request.getRequestId(), error);
                        return null;
                    });
        } catch (RejectedExecutionException e) {
            log.warn("Local standby cache metadata update queue is full, requestId={}", request.getRequestId());
        }
    }

    private void updateCacheMetadataNow(Request request,
                                        LocalStandbyHashResult hashResult,
                                        List<ServerStatus> selectedWorkers) {
        if (hashResult.blockCacheKeys().isEmpty() || hashResult.blockSize() <= 0) {
            return;
        }
        List<Long> cacheableBlockCacheKeys = request.getLocalStandbyCacheableBlockCacheKeys();
        if (cacheableBlockCacheKeys == null) {
            cacheableBlockCacheKeys = hashResult.blockCacheKeys();
        }
        if (cacheableBlockCacheKeys.isEmpty()) {
            return;
        }
        for (ServerStatus selectedWorker : selectedWorkers) {
            if (selectedWorker == null || !selectedWorker.isSuccess()) {
                continue;
            }
            RoleType workerRole = selectedWorker.getRole();
            if (workerRole != RoleType.PREFILL && workerRole != RoleType.PDFUSION) {
                continue;
            }
            cacheManager.addRoutedRequestBlocks(
                    selectedWorker.getServerIp() + ":" + selectedWorker.getHttpPort(),
                    cacheableBlockCacheKeys);
        }
    }

    private ThreadPoolExecutor createExecutor(int queueCapacity, String threadName) {
        return new ThreadPoolExecutor(
                4,
                10,
                0,
                TimeUnit.MILLISECONDS,
                new ArrayBlockingQueue<>(queueCapacity),
                runnable -> {
                    Thread thread = new Thread(runnable, threadName);
                    thread.setDaemon(true);
                    return thread;
                },
                new ThreadPoolExecutor.AbortPolicy());
    }

    @PreDestroy
    public void shutdown() {
        asyncMatchExecutor.shutdown();
        updateExecutor.shutdown();
    }
}
