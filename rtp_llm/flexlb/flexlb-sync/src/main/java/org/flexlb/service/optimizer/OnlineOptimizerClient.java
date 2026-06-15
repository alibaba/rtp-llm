package org.flexlb.service.optimizer;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.dao.optimizer.CommonResponseHeader;
import org.flexlb.dao.optimizer.OptimizerGetInstanceRequest;
import org.flexlb.dao.optimizer.OptimizerGetInstanceResponse;
import org.flexlb.dao.optimizer.OptimizerInstanceParams;
import org.flexlb.dao.optimizer.OptimizerRegisterRequest;
import org.flexlb.dao.optimizer.OptimizerRegisterResponse;
import org.flexlb.dao.optimizer.OptimizerRemoveInstanceRequest;
import org.flexlb.dao.optimizer.OptimizerRemoveInstanceResponse;
import org.flexlb.dao.optimizer.OptimizerTraceQueryRequest;
import org.flexlb.dao.optimizer.OptimizerTraceQueryResponse;
import org.flexlb.transport.GeneralHttpNettyService;
import org.flexlb.util.IdUtils;

import java.net.URI;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

@Slf4j
public class OnlineOptimizerClient {

    private final GeneralHttpNettyService httpService;
    private final OptimizerAddressResolver addressResolver;
    private final String instanceGroup;
    private final String basePath;
    private final int registerTimeoutMs;

    private volatile URI optimizerUri;
    private volatile String addressSnapshot = "";
    private volatile String instanceId;
    private volatile boolean registered = false;
    private final AtomicBoolean started = new AtomicBoolean(false);

    private final ScheduledExecutorService retryScheduler =
            Executors.newSingleThreadScheduledExecutor(r -> {
                Thread t = new Thread(r, "optimizer-register-retry");
                t.setDaemon(true);
                return t;
            });

    private static final long INITIAL_RETRY_DELAY_MS = 1000;
    private static final long MAX_RETRY_DELAY_MS = 30_000;
    private static final double BACKOFF_MULTIPLIER = 2.0;
    private static final long JITTER_BOUND_MS = 2000;

    public OnlineOptimizerClient(GeneralHttpNettyService httpService,
                                 OptimizerAddressResolver addressResolver,
                                 String instanceGroup,
                                 String basePath,
                                 int registerTimeoutMs) {
        this.httpService = httpService;
        this.addressResolver = addressResolver;
        this.instanceGroup = instanceGroup;
        this.basePath = stripTrailingSlash(basePath);
        this.registerTimeoutMs = registerTimeoutMs;
    }

    private static String stripTrailingSlash(String s) {
        if (s == null) return "";
        String r = s;
        while (r.endsWith("/")) {
            r = r.substring(0, r.length() - 1);
        }
        return r;
    }

    public void startRegistrationAsync(String instanceId, OptimizerInstanceParams params) {
        if (!started.compareAndSet(false, true)) {
            log.info("OnlineOptimizer registration already started, skip duplicate call");
            return;
        }
        this.instanceId = instanceId;
        safeSubmit(() -> attemptRegistration(instanceId, params, INITIAL_RETRY_DELAY_MS));
    }

    private void safeSubmit(Runnable task) {
        try {
            retryScheduler.submit(task);
        } catch (RejectedExecutionException e) {
            log.warn("OnlineOptimizer scheduler rejected task: {}", e.getMessage());
        }
    }

    private void safeSchedule(Runnable task, long delayMs) {
        try {
            retryScheduler.schedule(task, delayMs, TimeUnit.MILLISECONDS);
        } catch (RejectedExecutionException e) {
            log.warn("OnlineOptimizer scheduler rejected scheduled task: {}", e.getMessage());
        }
    }

    private void attemptRegistration(String instanceId, OptimizerInstanceParams params, long currentDelayMs) {
        if (registered) return;

        try {
            // Defer resolver start to async retry so listen failures do not block startup.
            if (!addressResolver.start()) {
                log.info("OnlineOptimizer address resolver not yet started, will retry");
                scheduleRetry(instanceId, params, currentDelayMs);
                return;
            }
            refreshUri();
            if (optimizerUri == null) {
                log.info("OnlineOptimizer address not yet resolved, will retry");
                scheduleRetry(instanceId, params, currentDelayMs);
                return;
            }

            boolean success = registerWithCheck(instanceId, params);
            if (success) {
                this.registered = true;
                log.info("OnlineOptimizer registration completed, traceQuery enabled");
                return;
            }
        } catch (Exception e) {
            log.warn("OnlineOptimizer registration attempt failed: {}", e.getMessage());
        }

        scheduleRetry(instanceId, params, currentDelayMs);
    }

    private void scheduleRetry(String instanceId, OptimizerInstanceParams params, long currentDelayMs) {
        long jitter = ThreadLocalRandom.current().nextLong(0, JITTER_BOUND_MS);
        long nextDelay = Math.min((long) (currentDelayMs * BACKOFF_MULTIPLIER), MAX_RETRY_DELAY_MS);
        long actualDelay = currentDelayMs + jitter;
        log.info("OnlineOptimizer will retry registration in {}ms", actualDelay);

        safeSchedule(
                () -> attemptRegistration(instanceId, params, nextDelay),
                actualDelay);
    }

    private boolean registerWithCheck(String instanceId, OptimizerInstanceParams params) {
        OptimizerGetInstanceResponse existing = findExistingInstance(instanceId);

        if (existing != null) {
            if (params.matchesRemote(existing)) {
                log.info("OnlineOptimizer instance already registered with matching params, skip");
                return true;
            }
            log.info("OnlineOptimizer instance exists but params differ, removing first");
            if (!removeInstance(instanceId)) {
                log.warn("OnlineOptimizer removeInstance failed, instanceId={}", instanceId);
                return false;
            }
        }

        return doRegister(instanceId, params);
    }

    private OptimizerGetInstanceResponse findExistingInstance(String instanceId) {
        OptimizerGetInstanceRequest req = new OptimizerGetInstanceRequest();
        req.setTraceId(IdUtils.fastUuid());
        req.setInstanceId(instanceId);

        OptimizerGetInstanceResponse resp = httpService.request(
                req, optimizerUri, basePath + "/getInstance",
                OptimizerGetInstanceResponse.class
        ).block(Duration.ofMillis(registerTimeoutMs));

        if (resp == null) {
            log.warn("OnlineOptimizer getInstance returned null, instanceId={}", instanceId);
            return null;
        }
        if (!isOkHeader(resp.getHeader())) {
            // Non-OK or malformed response: treat as not found, register flow will retry on its own check.
            log.info("OnlineOptimizer getInstance returned non-OK status code={}, instanceId={}",
                    extractStatusCode(resp.getHeader()), instanceId);
            return null;
        }
        return resp;
    }

    private boolean removeInstance(String instanceId) {
        OptimizerRemoveInstanceRequest req = new OptimizerRemoveInstanceRequest();
        req.setTraceId(IdUtils.fastUuid());
        req.setInstanceId(instanceId);

        OptimizerRemoveInstanceResponse resp = httpService.request(
                req, optimizerUri, basePath + "/removeInstance",
                OptimizerRemoveInstanceResponse.class
        ).block(Duration.ofMillis(registerTimeoutMs));

        if (resp == null) {
            log.warn("OnlineOptimizer removeInstance returned null, instanceId={}", instanceId);
            return false;
        }
        if (!isOkHeader(resp.getHeader())) {
            log.warn("OnlineOptimizer removeInstance failed, status={}, message={}, instanceId={}",
                    extractStatusCode(resp.getHeader()),
                    extractStatusMessage(resp.getHeader()), instanceId);
            return false;
        }
        log.info("OnlineOptimizer removeInstance success, instanceId={}", instanceId);
        return true;
    }

    private boolean doRegister(String instanceId, OptimizerInstanceParams params) {
        OptimizerRegisterRequest req = new OptimizerRegisterRequest();
        req.setTraceId(IdUtils.fastUuid());
        req.setInstanceGroup(instanceGroup);
        req.setInstanceId(instanceId);
        req.setBlockSize(params.getBlockSize());
        req.setLocationSpecInfos(params.getLocationSpecInfos());
        req.setLocationSpecGroups(params.getLocationSpecGroups());
        req.setLinearStep(params.getLinearStep());
        req.setFullGroupName(params.getFullGroupName());

        OptimizerRegisterResponse resp = httpService.request(
                req, optimizerUri, basePath + "/registerInstance",
                OptimizerRegisterResponse.class
        ).block(Duration.ofMillis(registerTimeoutMs));

        if (resp == null) {
            log.warn("OnlineOptimizer registerInstance returned null, instanceId={}", instanceId);
            return false;
        }
        if (!isOkHeader(resp.getHeader())) {
            // Strict: only code == 1 means success; malformed counts as failure to keep retrying.
            log.warn("OnlineOptimizer registerInstance failed, status={}, message={}, instanceId={}",
                    extractStatusCode(resp.getHeader()),
                    extractStatusMessage(resp.getHeader()), instanceId);
            return false;
        }
        log.info("OnlineOptimizer registerInstance success: instanceId={}", instanceId);
        return true;
    }

    public void traceQuery(long requestId, List<Long> blockCacheKeys) {
        if (!registered || blockCacheKeys == null || blockCacheKeys.isEmpty()) {
            return;
        }
        try {
            refreshUri();
            // Snapshot to local to avoid TOCTOU: another thread could clear optimizerUri
            // between the null check and the HTTP call.
            URI uri = optimizerUri;
            if (uri == null) {
                return;
            }

            OptimizerTraceQueryRequest req = new OptimizerTraceQueryRequest();
            req.setTraceId(String.valueOf(requestId));
            req.setInstanceId(instanceId);
            req.setBlockKeys(blockCacheKeys);

            httpService.request(req, uri, basePath + "/traceQuery",
                            OptimizerTraceQueryResponse.class)
                    .subscribe(
                            resp -> {},
                            err -> log.debug("OnlineOptimizer traceQuery error: {}", err.getMessage()));
        } catch (Throwable t) {
            log.debug("OnlineOptimizer traceQuery dispatch failed: {}", t.getMessage());
        }
    }

    public boolean isRegistered() {
        return registered;
    }

    public void shutdown() {
        retryScheduler.shutdown();
        try {
            if (!retryScheduler.awaitTermination(2, TimeUnit.SECONDS)) {
                retryScheduler.shutdownNow();
            }
        } catch (InterruptedException e) {
            retryScheduler.shutdownNow();
            Thread.currentThread().interrupt();
        }
        addressResolver.shutdown();
    }

    private synchronized void refreshUri() {
        List<String> addresses = addressResolver.getAddresses();
        if (addresses.isEmpty()) {
            // Resolver reports zero hosts: drop cached URI to avoid hitting a dead address.
            if (optimizerUri != null) {
                log.info("OnlineOptimizer addresses empty, clearing cached URI: {}", addressSnapshot);
                this.optimizerUri = null;
                this.addressSnapshot = "";
            }
            return;
        }
        String first = addresses.get(0);
        if (!first.equals(addressSnapshot)) {
            this.optimizerUri = URI.create("http://" + first);
            this.addressSnapshot = first;
            log.info("OnlineOptimizer address updated: {}", first);
        }
    }

    private static boolean isOkHeader(CommonResponseHeader header) {
        return header != null
                && header.getStatus() != null
                && header.getStatus().getCode() == 1;
    }

    private static Integer extractStatusCode(CommonResponseHeader header) {
        // null means header/status absent; avoids colliding with a real -1 from server.
        if (header == null || header.getStatus() == null) return null;
        return header.getStatus().getCode();
    }

    private static String extractStatusMessage(CommonResponseHeader header) {
        if (header == null || header.getStatus() == null) return null;
        return header.getStatus().getMessage();
    }
}
