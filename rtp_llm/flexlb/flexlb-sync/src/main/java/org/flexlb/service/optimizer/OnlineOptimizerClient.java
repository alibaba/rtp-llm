package org.flexlb.service.optimizer;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.dao.optimizer.CommonResponseHeader;
import org.flexlb.dao.optimizer.OptimizerErrorCode;
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
    private volatile String registeredAddress = "";
    private volatile String instanceId;
    private volatile OptimizerInstanceParams instanceParams;
    private final AtomicBoolean started = new AtomicBoolean(false);
    private final AtomicBoolean shutdown = new AtomicBoolean(false);

    // Guarded by this. A campaign generation prevents an old retry or a slow HTTP
    // response from mutating a newer registration attempt after discovery moves.
    private boolean registrationInProgress;
    private long registrationCampaign;
    private String registrationTargetAddress = "";
    private long registrationEpoch;

    // Registration and retry timing share one daemon scheduler. Registration HTTP
    // may block this background thread up to registerTimeoutMs, but never blocks the
    // startup or request thread.
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

    private static final String PATH_GET_INSTANCE = "/getInstance";
    private static final String PATH_REGISTER_INSTANCE = "/registerInstance";
    private static final String PATH_REMOVE_INSTANCE = "/removeInstance";
    private static final String PATH_TRACE_QUERY = "/traceQuery";

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
        if (shutdown.get()) {
            log.info("OnlineOptimizer client already shutdown, skip registration");
            return;
        }
        if (!started.compareAndSet(false, true)) {
            log.info("OnlineOptimizer registration already started, skip duplicate call");
            return;
        }
        this.instanceId = instanceId;
        this.instanceParams = params;
        ensureRegistrationForTarget(addressSnapshot);
    }

    private boolean safeSubmit(Runnable task) {
        try {
            retryScheduler.submit(task);
            return true;
        } catch (RejectedExecutionException e) {
            log.warn("OnlineOptimizer scheduler rejected task: {}", e.getMessage());
            return false;
        }
    }

    private boolean safeSchedule(Runnable task, long delayMs) {
        try {
            retryScheduler.schedule(task, delayMs, TimeUnit.MILLISECONDS);
            return true;
        } catch (RejectedExecutionException e) {
            log.warn("OnlineOptimizer scheduler rejected scheduled task: {}", e.getMessage());
            return false;
        }
    }

    private synchronized void ensureRegistrationForTarget(String targetAddress) {
        if (shutdown.get() || instanceId == null || instanceParams == null) {
            return;
        }
        String desiredAddress = targetAddress == null ? "" : targetAddress;
        if (!registeredAddress.isEmpty() && desiredAddress.equals(registeredAddress)) {
            return;
        }
        if (registrationInProgress && desiredAddress.equals(registrationTargetAddress)) {
            return;
        }

        long campaign = ++registrationCampaign;
        registrationInProgress = true;
        registrationTargetAddress = desiredAddress;
        if (!safeSubmit(() -> attemptRegistration(campaign, INITIAL_RETRY_DELAY_MS))) {
            finishRegistrationCampaign(campaign);
        }
    }

    private void attemptRegistration(long campaign, long currentDelayMs) {
        if (!isRegistrationCampaignCurrent(campaign)) {
            return;
        }

        try {
            // Defer resolver start to async retry so discovery I/O does not block startup.
            if (!addressResolver.start()) {
                log.info("OnlineOptimizer address resolver not yet started, will retry");
                scheduleRetry(campaign, currentDelayMs);
                return;
            }
            ResolvedTarget target = refreshAndSnapshotTarget();
            URI targetUri = target.uri();
            String targetAddress = target.address();
            if (!bindRegistrationCampaignTarget(campaign, targetAddress)) {
                return;
            }
            if (targetUri == null || targetAddress.isEmpty()) {
                log.info("OnlineOptimizer address not yet resolved, will retry");
                scheduleRetry(campaign, currentDelayMs);
                return;
            }

            boolean success = registerWithCheck(instanceId, instanceParams, targetUri);
            if (success && markRegisteredIfCurrent(campaign, targetAddress)) {
                finishRegistrationCampaign(campaign);
                log.info("OnlineOptimizer registration completed, traceQuery enabled");
                return;
            }
        } catch (Exception e) {
            log.warn("OnlineOptimizer registration attempt failed: {}", e.getMessage());
        }

        scheduleRetry(campaign, currentDelayMs);
    }

    private void scheduleRetry(long campaign, long currentDelayMs) {
        if (!isRegistrationCampaignCurrent(campaign)) {
            return;
        }
        long jitter = ThreadLocalRandom.current().nextLong(0, JITTER_BOUND_MS);
        long nextDelay = Math.min((long) (currentDelayMs * BACKOFF_MULTIPLIER), MAX_RETRY_DELAY_MS);
        long actualDelay = currentDelayMs + jitter;
        log.info("OnlineOptimizer will retry registration in {}ms", actualDelay);

        if (!safeSchedule(
                () -> attemptRegistration(campaign, nextDelay),
                actualDelay)) {
            finishRegistrationCampaign(campaign);
        }
    }

    private synchronized boolean isRegistrationCampaignCurrent(long campaign) {
        if (shutdown.get()
                || !registrationInProgress
                || campaign != registrationCampaign) {
            return false;
        }
        if (!registeredAddress.isEmpty()) {
            finishRegistrationCampaign(campaign);
            return false;
        }
        return true;
    }

    private synchronized boolean bindRegistrationCampaignTarget(long campaign, String targetAddress) {
        if (shutdown.get()
                || !registrationInProgress
                || campaign != registrationCampaign) {
            return false;
        }
        registrationTargetAddress = targetAddress == null ? "" : targetAddress;
        return true;
    }

    private synchronized void finishRegistrationCampaign(long campaign) {
        if (campaign == registrationCampaign) {
            registrationInProgress = false;
            registrationTargetAddress = "";
        }
    }

    private boolean registerWithCheck(String instanceId, OptimizerInstanceParams params, URI targetUri) {
        OptimizerGetInstanceResponse existing = findExistingInstance(instanceId, targetUri);

        if (existing != null) {
            if (params.matchesRemote(existing)) {
                log.info("OnlineOptimizer instance already registered with matching params, skip");
                return true;
            }
            log.info("OnlineOptimizer instance exists but params differ, removing first");
            if (!removeInstance(instanceId, targetUri)) {
                log.warn("OnlineOptimizer removeInstance failed, instanceId={}", instanceId);
                return false;
            }
        }

        OptimizerErrorCode registerStatus = doRegister(instanceId, params, targetUri);
        if (registerStatus == OptimizerErrorCode.OK) {
            return true;
        }
        if (registerStatus == OptimizerErrorCode.DUPLICATE_ENTITY) {
            // The latest official client permits duplicate registration. Verify the
            // winner of the race before treating this client as registered, so an
            // instance-id collision with different params cannot corrupt TraceQuery.
            OptimizerGetInstanceResponse racedExisting = findExistingInstance(instanceId, targetUri);
            if (racedExisting != null && params.matchesRemote(racedExisting)) {
                log.info("OnlineOptimizer duplicate registration has matching params, instanceId={}",
                        instanceId);
                return true;
            }
            log.warn("OnlineOptimizer duplicate registration has mismatched params, instanceId={}",
                    instanceId);
        }
        return false;
    }

    private OptimizerGetInstanceResponse findExistingInstance(String instanceId, URI targetUri) {
        OptimizerGetInstanceRequest req = new OptimizerGetInstanceRequest();
        req.setTraceId(IdUtils.fastUuid());
        req.setInstanceId(instanceId);

        OptimizerGetInstanceResponse resp = httpService.request(
                req, targetUri, basePath + PATH_GET_INSTANCE,
                OptimizerGetInstanceResponse.class
        ).block(Duration.ofMillis(registerTimeoutMs));

        if (resp == null) {
            throw new IllegalStateException(
                    "OnlineOptimizer getInstance returned null, instanceId=" + instanceId);
        }
        OptimizerErrorCode statusCode = extractStatusCode(resp.getHeader());
        if (statusCode == OptimizerErrorCode.INSTANCE_NOT_EXIST) {
            log.info("OnlineOptimizer instance does not exist, instanceId={}", instanceId);
            return null;
        }
        if (statusCode != OptimizerErrorCode.OK) {
            throw new IllegalStateException(
                    "OnlineOptimizer getInstance failed, status=" + statusCode
                            + ", message=" + extractStatusMessage(resp.getHeader())
                            + ", instanceId=" + instanceId);
        }
        return resp;
    }

    private synchronized boolean markRegisteredIfCurrent(long campaign, String targetAddress) {
        if (shutdown.get()
                || !registrationInProgress
                || campaign != registrationCampaign
                || !targetAddress.equals(addressSnapshot)) {
            return false;
        }
        registeredAddress = targetAddress;
        registrationEpoch++;
        return true;
    }

    private boolean removeInstance(String instanceId, URI targetUri) {
        OptimizerRemoveInstanceRequest req = new OptimizerRemoveInstanceRequest();
        req.setTraceId(IdUtils.fastUuid());
        req.setInstanceId(instanceId);

        OptimizerRemoveInstanceResponse resp = httpService.request(
                req, targetUri, basePath + PATH_REMOVE_INSTANCE,
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

    private OptimizerErrorCode doRegister(
            String instanceId,
            OptimizerInstanceParams params,
            URI targetUri) {
        OptimizerRegisterRequest req = new OptimizerRegisterRequest();
        req.setTraceId(IdUtils.fastUuid());
        req.setInstanceGroup(instanceGroup);
        req.setInstanceId(instanceId);
        req.setBlockSize(params.getBlockSize());
        req.setLocationSpecInfos(params.getLocationSpecInfos());
        req.setLocationSpecGroups(params.getLocationSpecGroups());
        req.setOptimizerStateInfo(params.getOptimizerStateInfo());
        req.setLinearStep(params.getLinearStep());

        OptimizerRegisterResponse resp = httpService.request(
                req, targetUri, basePath + PATH_REGISTER_INSTANCE,
                OptimizerRegisterResponse.class
        ).block(Duration.ofMillis(registerTimeoutMs));

        if (resp == null) {
            log.warn("OnlineOptimizer registerInstance returned null, instanceId={}", instanceId);
            return null;
        }
        OptimizerErrorCode statusCode = extractStatusCode(resp.getHeader());
        if (statusCode != OptimizerErrorCode.OK
                && statusCode != OptimizerErrorCode.DUPLICATE_ENTITY) {
            log.warn("OnlineOptimizer registerInstance failed, status={}, message={}, instanceId={}",
                    statusCode,
                    extractStatusMessage(resp.getHeader()), instanceId);
            return statusCode;
        }
        log.info("OnlineOptimizer registerInstance returned status={}: instanceId={}",
                statusCode, instanceId);
        return statusCode;
    }

    public void traceQuery(String requestId, List<Long> blockCacheKeys, long inputTokenLen) {
        if (shutdown.get() || blockCacheKeys == null || blockCacheKeys.isEmpty()) {
            return;
        }
        try {
            ResolvedTarget target = refreshAndSnapshotTarget();
            if (!target.registered()) {
                ensureRegistrationForTarget(target.address());
                return;
            }
            URI uri = target.uri();
            String targetAddress = target.address();

            OptimizerTraceQueryRequest req = new OptimizerTraceQueryRequest();
            req.setTraceId(requestId);
            req.setInstanceId(instanceId);
            req.setBlockKeys(blockCacheKeys);
            req.setTokenIds(List.of());
            req.setInputTokenLen(inputTokenLen);

            httpService.request(
                            req, uri, basePath + PATH_TRACE_QUERY,
                            OptimizerTraceQueryResponse.class)
                    .subscribe(
                            resp -> handleTraceQueryResponse(
                                    targetAddress, target.registrationEpoch(), resp),
                            err -> log.debug("OnlineOptimizer traceQuery error: {}", err.getMessage()));
        } catch (Throwable t) {
            log.debug("OnlineOptimizer traceQuery dispatch failed: {}", t.getMessage());
        }
    }

    private void handleTraceQueryResponse(
            String targetAddress,
            long targetRegistrationEpoch,
            OptimizerTraceQueryResponse response) {
        OptimizerErrorCode statusCode = extractStatusCode(response.getHeader());
        if (statusCode == OptimizerErrorCode.OK) {
            return;
        }
        if (statusCode == OptimizerErrorCode.INSTANCE_NOT_EXIST
                && invalidateRegistrationIfCurrent(targetAddress, targetRegistrationEpoch)) {
            log.info("OnlineOptimizer traceQuery reports missing instance; re-registering: address={}",
                    targetAddress);
            ensureRegistrationForTarget(targetAddress);
            return;
        }
        log.debug("OnlineOptimizer traceQuery returned non-OK status: address={}, status={}",
                targetAddress, statusCode);
    }

    private synchronized boolean invalidateRegistrationIfCurrent(
            String targetAddress,
            long targetRegistrationEpoch) {
        if (registeredAddress.isEmpty()
                || targetRegistrationEpoch != registrationEpoch
                || !targetAddress.equals(registeredAddress)) {
            return false;
        }
        clearRegistrationState();
        return true;
    }

    public boolean isRegistered() {
        return !registeredAddress.isEmpty();
    }

    public void shutdown() {
        if (!shutdown.compareAndSet(false, true)) {
            return;
        }
        // Intentionally NOT calling removeInstance: keep the registration alive on the
        // optimizer server so that a restarted instance can reuse the same slot without
        // re-registration overhead (the server will match by instanceId on next startup).
        synchronized (this) {
            registrationCampaign++;
            registrationInProgress = false;
            registrationTargetAddress = "";
            clearRegistrationState();
        }
        retryScheduler.shutdownNow();
        try {
            retryScheduler.awaitTermination(2, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            retryScheduler.shutdownNow();
            Thread.currentThread().interrupt();
        }
        addressResolver.shutdown();
    }

    private synchronized ResolvedTarget refreshAndSnapshotTarget() {
        List<String> addresses = addressResolver.getAddresses();
        if (addresses.isEmpty()) {
            // Resolver reports zero hosts: drop cached URI to avoid hitting a dead address.
            if (optimizerUri != null) {
                log.info("OnlineOptimizer addresses empty, clearing cached URI: {}", addressSnapshot);
                this.optimizerUri = null;
                this.addressSnapshot = "";
            }
            clearRegistrationState();
            return snapshotTarget();
        }
        // Keep the selected stateful optimizer while it remains healthy. A discovery-only
        // ordering change must not move TraceQuery traffic to an unregistered replica.
        String selected = addresses.contains(addressSnapshot) ? addressSnapshot : addresses.get(0);
        if (!selected.equals(addressSnapshot)) {
            this.optimizerUri = URI.create("http://" + selected);
            this.addressSnapshot = selected;
            if (!selected.equals(registeredAddress)) {
                clearRegistrationState();
            }
            log.info("OnlineOptimizer address updated: {}", selected);
        }
        return snapshotTarget();
    }

    private ResolvedTarget snapshotTarget() {
        return new ResolvedTarget(
                optimizerUri,
                addressSnapshot,
                !registeredAddress.isEmpty() && addressSnapshot.equals(registeredAddress),
                registrationEpoch);
    }

    private void clearRegistrationState() {
        if (!registeredAddress.isEmpty()) {
            registrationEpoch++;
        }
        registeredAddress = "";
    }

    private record ResolvedTarget(
            URI uri,
            String address,
            boolean registered,
            long registrationEpoch) {}

    private static boolean isOkHeader(CommonResponseHeader header) {
        return header != null
                && header.getStatus() != null
                && header.getStatus().isOk();
    }

    private static OptimizerErrorCode extractStatusCode(CommonResponseHeader header) {
        if (header == null || header.getStatus() == null) return null;
        return header.getStatus().getCode();
    }

    private static String extractStatusMessage(CommonResponseHeader header) {
        if (header == null || header.getStatus() == null) return null;
        return header.getStatus().getMessage();
    }
}
