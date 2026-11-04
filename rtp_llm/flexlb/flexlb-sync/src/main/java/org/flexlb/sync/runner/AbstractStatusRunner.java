package org.flexlb.sync.runner;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.dao.master.EndpointTrace.Stage;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.grpc.WorkerStatusRpcClient;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.util.IdUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Objects;
import java.util.concurrent.Executor;
import java.util.concurrent.TimeUnit;

/** Starts one poll; the response workflow owns its lease after RPC initiation succeeds. */
abstract class AbstractStatusRunner implements Runnable {
    protected static final Logger logger = LoggerFactory.getLogger("syncLogger");
    protected final String ipPort;
    protected final String modelName;
    protected final RoleType roleType;
    protected final WorkerStatus workerStatus;
    protected final WorkerStatus.PollLease pollLease;
    protected final EndpointRegistry endpointRegistry;
    protected final EngineHealthReporter engineHealthReporter;
    protected final WorkerStatusRpcClient workerStatusRpcClient;
    protected final CacheAwareService cacheAwareService;
    protected final long requestTimeoutMs;
    protected final Executor callbackExecutor;
    protected final long createTimeUs = TimeUnit.NANOSECONDS.toMicros(System.nanoTime());
    protected final String id = IdUtils.fastUuid();

    protected AbstractStatusRunner(
            String modelName,
            WorkerStatus workerStatus,
            WorkerStatus.PollLease pollLease,
            EndpointRegistry endpointRegistry,
            EngineHealthReporter engineHealthReporter,
            WorkerStatusRpcClient workerStatusRpcClient,
            CacheAwareService cacheAwareService,
            long requestTimeoutMs,
            Executor callbackExecutor) {
        this.ipPort = workerStatus.getLogicalIpPort();
        this.modelName = modelName;
        this.roleType = workerStatus.getRole();
        this.workerStatus = workerStatus;
        this.pollLease = Objects.requireNonNull(pollLease, "pollLease");
        this.endpointRegistry = Objects.requireNonNull(endpointRegistry, "endpointRegistry");
        this.engineHealthReporter = engineHealthReporter;
        this.workerStatusRpcClient = workerStatusRpcClient;
        this.cacheAwareService = cacheAwareService;
        this.requestTimeoutMs = requestTimeoutMs;
        this.callbackExecutor = callbackExecutor;
    }

    @Override
    public final void run() {
        pollLease.trace().record(Stage.RUNNER_STARTED);
        boolean asyncInitiated = false;
        try {
            startRpc();
            asyncInitiated = true;
        } catch (RuntimeException | Error failure) {
            pollLease.trace().record(Stage.RPC_INITIATION_FAILED, failure);
            throw failure;
        } finally {
            if (!asyncInitiated) {
                pollLease.close();
            }
        }
    }

    protected abstract void startRpc();

    protected final void handleException(Throwable failure) {
        boolean timeout =
                io.grpc.Status.fromThrowable(failure).getCode()
                        == io.grpc.Status.Code.DEADLINE_EXCEEDED;
        if (logger.isDebugEnabled()) {
            logger.debug(
                    "gRPC poll failed: runner={} id={} site={} endpoint={} model={} elapsed_us={}"
                            + " timeout={} generation={} trace={}",
                    getClass().getSimpleName(),
                    id,
                    workerStatus.getSite(),
                    ipPort,
                    modelName,
                    TimeUnit.NANOSECONDS.toMicros(System.nanoTime()) - createTimeUs,
                    timeout,
                    workerStatus.getGenerationId(),
                    workerStatus.trace().snapshot().compact(),
                    failure);
        }
        reportRpcFailure(timeout);
    }

    protected abstract void reportRpcFailure(boolean timeout);
}
