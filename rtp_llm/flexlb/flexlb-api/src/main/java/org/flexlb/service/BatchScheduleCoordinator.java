package org.flexlb.service;

import com.google.common.util.concurrent.RateLimiter;
import org.flexlb.balance.strategy.RoundRobinLoadBalancer;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.loadbalance.BatchScheduleRequest;
import org.flexlb.dao.loadbalance.BatchScheduleResponse;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dispatcher.FePool;
import org.flexlb.exception.BatchScheduleTransportException;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.ClientResponse;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.net.URI;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.TimeoutException;

/** Allocate on the elected master or forward once; failed forwarding never allocates locally. */
@Component
public class BatchScheduleCoordinator {

    private static final int MAX_FORWARD_HOPS = 1;

    private static final Duration REQUEST_TIMEOUT = Duration.ofSeconds(3);

    private final ObjectProvider<FePool> fePoolProvider;
    private final RoundRobinLoadBalancer batchScheduler;
    private final LBStatusConsistencyService consistency;
    private final WebClient webClient;
    private final EngineHealthReporter engineHealthReporter;
    private final FlexlbConfig config;
    /** A dead/unknown master fails every forwarded batch request; cap the ERROR stream at 1/s. */
    private final RateLimiter masterUnreachableWarn = RateLimiter.create(1);

    public BatchScheduleCoordinator(RoundRobinLoadBalancer batchScheduler,
                                    LBStatusConsistencyService consistency,
                                    WebClient.Builder webClientBuilder,
                                    EngineHealthReporter engineHealthReporter,
                                    ObjectProvider<FePool> fePoolProvider,
                                    ConfigService configService) {
        this.fePoolProvider = fePoolProvider;
        this.batchScheduler = batchScheduler;
        this.consistency = consistency;
        this.webClient = webClientBuilder.clone()
                .codecs(codecs -> codecs.defaultCodecs().maxInMemorySize(16 * 1024 * 1024))
                .build();
        this.engineHealthReporter = engineHealthReporter;
        this.config = configService.loadBalanceConfig();
    }

    public Mono<BatchScheduleResponse> schedule(BatchScheduleRequest request) {
        return Mono.defer(() -> {
            int maxCount = config.getRouter().getBatchScheduleMaxCount();
            if (request == null || request.getBatchCount() < 1 || request.getBatchCount() > maxCount) {
                return Mono.just(BatchScheduleResponse.error(StrategyErrorType.INVALID_REQUEST,
                        "batch_count must be in [1, " + maxCount + "]"));
            }
            if (request.getAllocationType() == null) {
                return Mono.just(BatchScheduleResponse.error(StrategyErrorType.INVALID_REQUEST,
                        "allocation_type must be BE, FE or FE_AND_BE"));
            }
            if (consistency.isNeedConsistency() && !consistency.isMaster()) {
                return forwardToMaster(request);
            }
            return resolveLocally(request, consistency.getMasterHostIpPort());
        })
                .switchIfEmpty(Mono.error(new BatchScheduleTransportException(
                        "empty batch scheduling response", "EMPTY_RESPONSE")))
                .map(response -> validateTargets(request, response))
                .timeout(REQUEST_TIMEOUT)
                .onErrorMap(TimeoutException.class,
                        error -> new BatchScheduleTransportException("batch scheduling timed out", "TIMEOUT"));
    }

    private static BatchScheduleResponse validateTargets(
            BatchScheduleRequest request, BatchScheduleResponse response) {
        if (!response.isSuccess()) {
            return response;
        }
        List<String> urls = response.getFrontendUrls();
        boolean valid = !request.getAllocationType().includesFe() || (urls != null && urls.size() == request.getBatchCount()
                && urls.stream().allMatch(url -> url != null && !url.isBlank()));
        List<BatchScheduleTarget> targets = response.getServerStatus();
        if (request.getAllocationType().includesBe()) {
            valid &= targets != null && targets.size() == request.getBatchCount();
        }
        if (valid && request.getAllocationType().includesBe()) {
            for (BatchScheduleTarget target : targets) {
                if (target == null
                        || target.getRole() == null
                        || target.getServerIp() == null || target.getServerIp().isBlank()
                        || target.getHttpPort() <= 0
                        || !((target.getGrpcPort() != null && target.getGrpcPort() > 0)
                            ^ (target.getArpcPort() != null && target.getArpcPort() > 0))) {
                    valid = false;
                    break;
                }
            }
        }
        return valid ? response : BatchScheduleResponse.error(StrategyErrorType.NO_AVAILABLE_WORKER,
                "batch scheduling returned incomplete targets");
    }

    private Mono<BatchScheduleResponse> forwardToMaster(BatchScheduleRequest request) {
        String master = consistency.getMasterHostIpPort();
        if (master == null) {
            if (masterUnreachableWarn.tryAcquire()) {
                Logger.warn("[BatchSchedule] Master unreachable: no elected master");
            }
            engineHealthReporter.reportForwardToMasterResult("LOCAL", "MASTER_NULL");
            return Mono.error(new BatchScheduleTransportException(
                    "master unreachable", "MASTER_NULL"));
        }
        URI uri = URI.create("http://" + master);
        // Match the gRPC forwarder's guard: stale leader views must not form relay loops.
        long incomingHop = Integer.toUnsignedLong(request.getForwardHop());
        String blockedReason = incomingHop >= MAX_FORWARD_HOPS
                ? "FORWARD_HOP_LIMIT"
                : uri.getHost() != null && uri.getHost().equals(consistency.getLocalHostIp())
                        ? "SELF_FORWARD_BLOCKED"
                        : null;
        if (blockedReason != null) {
            if (masterUnreachableWarn.tryAcquire()) {
                Logger.warn("[BatchSchedule] Forward blocked: reason={}, master={}, hop={}",
                        blockedReason, master, incomingHop);
            }
            engineHealthReporter.reportForwardToMasterResult(uri.getHost(), blockedReason);
            return Mono.error(new BatchScheduleTransportException(
                    "batch schedule forward blocked: " + blockedReason, blockedReason));
        }

        BatchScheduleRequest forwarded = new BatchScheduleRequest();
        forwarded.setBatchCount(request.getBatchCount());
        forwarded.setAllocationType(request.getAllocationType());
        forwarded.setForwardHop(Math.toIntExact(incomingHop + 1));
        Logger.debug("[BatchSchedule] Forwarding to master {}: batchCount={}", master, request.getBatchCount());
        return webClient.post().uri(uri.resolve("/rtp_llm/batch_schedule"))
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(forwarded)
                .exchangeToMono(this::readMasterResponse)
                .onErrorMap(error -> error instanceof BatchScheduleTransportException ? error
                        : new BatchScheduleTransportException("master unreachable", "CONNECT_FAILED"))
                .doOnNext(response -> engineHealthReporter.reportForwardToMasterResult(
                        uri.getHost(), String.valueOf(response.getCode())))
                .doOnError(BatchScheduleTransportException.class, error -> {
                    if (masterUnreachableWarn.tryAcquire()) {
                        Logger.warn("[BatchSchedule] Forward failed: master={}, errorCode={}",
                                master, error.getErrorCode());
                    }
                    engineHealthReporter.reportForwardToMasterResult(uri.getHost(), error.getErrorCode());
                });
    }

    private Mono<BatchScheduleResponse> readMasterResponse(ClientResponse httpResponse) {
        boolean successStatus = httpResponse.statusCode().is2xxSuccessful();
        String errorCode = successStatus ? "INVALID_RESPONSE" : "HTTP_ERROR";
        return httpResponse.bodyToMono(BatchScheduleResponse.class)
                .filter(response -> successStatus || (httpResponse.statusCode().isError()
                        && !response.isSuccess() && response.getErrorMessage() != null))
                .switchIfEmpty(Mono.error(new BatchScheduleTransportException(
                        "invalid master response", errorCode)))
                .onErrorMap(error -> error instanceof BatchScheduleTransportException ? error
                        : new BatchScheduleTransportException("invalid master response", errorCode));
    }

    private Mono<BatchScheduleResponse> resolveLocally(BatchScheduleRequest request,
                                                       String electedMaster) {
        // FE-only dispatch never enters worker placement or depends on BE topology/readiness.
        Mono<BatchScheduleResponse> workers = request.getAllocationType().includesBe()
                ? batchScheduler.schedule(request.getBatchCount())
                : Mono.just(BatchScheduleResponse.success(List.of()));
        return workers
                .map(response -> {
                    BatchScheduleResponse completed = response;
                    if (response.isSuccess() && request.getAllocationType().includesFe()) {
                        completed = assignFrontends(response, request.getBatchCount());
                    }
                    completed.setRealMasterHost(electedMaster);
                    return completed;
                });
    }

    private BatchScheduleResponse assignFrontends(BatchScheduleResponse response, int count) {
        FePool pool = config.getHttpDispatcher().isEnabled() ? fePoolProvider.getIfAvailable() : null;
        if (pool == null) {
            return BatchScheduleResponse.error(StrategyErrorType.NO_AVAILABLE_WORKER,
                    "no FE pool configured");
        }
        try {
            response.setFrontendUrls(pool.nextBatch(count));
        } catch (IllegalStateException emptyPool) {
            return BatchScheduleResponse.error(StrategyErrorType.NO_AVAILABLE_WORKER,
                    "no FE endpoints available");
        }
        return response;
    }
}
