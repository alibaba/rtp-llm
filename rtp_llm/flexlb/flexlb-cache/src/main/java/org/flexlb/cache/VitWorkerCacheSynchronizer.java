package org.flexlb.cache;

import org.flexlb.config.service.RoleType;
import org.flexlb.master.MasterStatusView;
import org.flexlb.util.Logger;
import org.flexlb.worker.endpoint.EndpointRegistry;
import org.flexlb.worker.rpc.EngineGrpcClient;
import org.flexlb.worker.rpc.protocol.EngineRpcService.CacheVersionPB;
import org.flexlb.worker.status.WorkerStatus;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/**
 * Polls ViT cache residency on standalone nodes or the elected master and publishes snapshots
 * to the cache manager. Becoming eligible again refreshes immediately; shutdown interrupts polling.
 */
@Component
class VitWorkerCacheSynchronizer {
    // Cache residency changes after each image; ten-minute snapshots lose affinity.
    static final long SYNC_INTERVAL_MS = 30_000;
    private final EndpointRegistry workers;
    private final EngineGrpcClient grpc;
    private final MasterStatusView masterStatus;
    private final VitWorkerCacheManager cacheManager;
    private final Map<WorkerStatus, Long> attempts = new IdentityHashMap<>();
    private final ScheduledExecutorService sync =
            Executors.newSingleThreadScheduledExecutor(
                    r -> {
                        Thread thread = new Thread(r, "vit-cache-sync");
                        thread.setDaemon(true);
                        return thread;
                    });
    private boolean syncEnabledLastRound;

    VitWorkerCacheSynchronizer(
            EndpointRegistry workers,
            EngineGrpcClient grpc,
            MasterStatusView masterStatus,
            VitWorkerCacheManager cacheManager) {
        this.workers = workers;
        this.grpc = grpc;
        this.masterStatus = masterStatus;
        this.cacheManager = cacheManager;
    }

    @PostConstruct
    void start() {
        sync.scheduleWithFixedDelay(this::refresh, 0, 5, TimeUnit.SECONDS);
    }

    @PreDestroy
    void stop() {
        sync.shutdownNow();
    }

    void refresh() {
        if (sync.isShutdown()) return;
        try {
            if (masterStatus.isNeedConsistency() && !masterStatus.isMaster()) {
                syncEnabledLastRound = false;
                return;
            }
            Map<String, WorkerStatus> live = liveWorkers();
            List<WorkerStatus> due = new ArrayList<>();
            long now = System.currentTimeMillis();
            cacheManager.prune(live, now);
            attempts.keySet().removeIf(worker -> live.get(worker.getLogicalIpPort()) != worker);
            if (!syncEnabledLastRound) {
                attempts.clear();
            }
            for (WorkerStatus worker : live.values()) {
                if (now - attempts.getOrDefault(worker, 0L) >= SYNC_INTERVAL_MS) {
                    attempts.put(worker, now);
                    due.add(worker);
                }
            }
            syncEnabledLastRound = true;
            CacheVersionPB request = CacheVersionPB.newBuilder().setNeedCacheKeys(true).build();
            Flux.fromIterable(due)
                    .flatMap(
                            worker ->
                                    Mono.defer(
                                                    () ->
                                                            Mono.fromFuture(
                                                                    grpc
                                                                            .getMultimodalCacheStatusAsync(
                                                                                    worker.getIp(),
                                                                                    worker
                                                                                            .getWorkerStatusPort(),
                                                                                    request,
                                                                                    2000)))
                                            .filter(response -> response.hasMultimodalCache())
                                            .doOnNext(
                                                    response ->
                                                            cacheManager.replace(
                                                                    worker,
                                                                    response.getMultimodalCache()))
                                            .onErrorResume(
                                                    error -> {
                                                        Logger.debug(
                                                                "ViT cache snapshot unavailable for {}: {}",
                                                                worker.getLogicalIpPort(),
                                                                error.toString());
                                                        return Mono.empty();
                                                    }),
                            4)
                    .then()
                    .block();
        } catch (Exception error) {
            Logger.warn("ViT cache sync failed: {}", error.toString());
        }
    }

    private Map<String, WorkerStatus> liveWorkers() {
        Map<String, WorkerStatus> live = new HashMap<>();
        for (var endpoint : workers.endpoints(RoleType.VIT, null)) {
            if (!endpoint.acceptsNewRequests()) continue;
            String address = endpoint.ipPort();
            if (workers.getEndpoint(RoleType.VIT, address) == endpoint) {
                live.put(address, endpoint.getStatus());
            }
        }
        return live;
    }
}
