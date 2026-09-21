package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.TrafficPolicyConfig;
import org.flexlb.dao.loadbalance.BatchScheduleRequest;
import org.flexlb.dao.loadbalance.BatchScheduleResponse;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.enums.EngineType;
import org.flexlb.exception.BatchScheduleTransportException;
import org.flexlb.service.BatchScheduleCoordinator;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.core.io.buffer.DataBufferLimitException;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Scheduler;

import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import static org.flexlb.dispatcher.FanoutService.MAX_AGGREGATE_BYTES;

@Component
@ConditionalOnProperty(prefix = "dispatch", name = "fe-pool-service-id")
public class BatchHandler {

    private final DispatchConfig cfg;
    private final FanoutService fanoutService;
    private final BatchScheduleCoordinator batchScheduleCoordinator;
    private final PassthroughClient passthroughClient;
    private final DispatcherMetricsReporter metricsReporter;
    private final FlexlbConfig loadBalanceConfig;
    private final Scheduler cpuScheduler;

    public BatchHandler(FanoutService fanoutService,
                        DispatchConfig cfg,
                        BatchScheduleCoordinator batchScheduleCoordinator,
                        PassthroughClient passthroughClient,
                        DispatcherMetricsReporter metricsReporter,
                        ConfigService configService,
                        @Qualifier("dispatcherCpuScheduler") Scheduler cpuScheduler) {
        this.cfg = cfg;
        this.fanoutService = fanoutService;
        this.batchScheduleCoordinator = batchScheduleCoordinator;
        this.passthroughClient = passthroughClient;
        this.metricsReporter = metricsReporter;
        this.loadBalanceConfig = configService.loadBalanceConfig();
        this.cpuScheduler = cpuScheduler;
    }

    public Mono<ServerResponse> handle(ServerRequest request, BatchEndpointSpec spec, boolean dryRun) {
        long start = System.currentTimeMillis();
        AtomicInteger status = new AtomicInteger(499);
        AtomicBoolean delegatedToPassthrough = new AtomicBoolean(false);
        return request.bodyToMono(byte[].class).defaultIfEmpty(new byte[0])
                .flatMap(bytes -> Mono.defer(
                                () -> handleBody(request, spec, bytes, dryRun, delegatedToPassthrough))
                        .subscribeOn(cpuScheduler))
                .onErrorResume(e -> {
                    String errMsg = e.toString();
                    Logger.warn("dispatcher request failed: spec={}, err={}", spec.getPath(), errMsg);
                    if (e instanceof BatchScheduleTransportException) {
                        return DispatcherResponses.error(503, "batch_schedule_failed", "batch target allocation failed");
                    }
                    if (e instanceof DataBufferLimitException) {
                        return DispatcherResponses.error(413, "request_body_too_large",
                                "batch body exceeds the server limit; see MAX_IN_MEMORY_SIZE");
                    }
                    if (e instanceof ResponseTooLargeException) {
                        return DispatcherResponses.error(413, "batch_response_too_large",
                                "sub-batch responses exceed the dispatcher byte limit");
                    }
                    if (e instanceof AggregateRequestTooLargeException) {
                        return DispatcherResponses.error(413, "batch_request_too_large",
                                "aggregate sub-batch request exceeds the dispatcher limit");
                    }
                    // Upstream exception text stays in logs, outside the client response.
                    return DispatcherResponses.error(500, "dispatch_failed", "batch dispatch failed");
                })
                .doOnNext(response -> status.set(response.rawStatusCode()))
                .doFinally(signal -> {
                    if (!dryRun && !delegatedToPassthrough.get()) {
                        metricsReporter.reportRequest("batch", spec.getPath(), status.get(), System.currentTimeMillis() - start);
                    }
                });
    }

    private Mono<ServerResponse> handleBody(ServerRequest request, BatchEndpointSpec spec, byte[] bytes,
                                          boolean dryRun, AtomicBoolean delegatedToPassthrough) {
        JSONObject body = BatchBodyParser.parseObject(bytes);
        if (body == null) {
            return badRequest("expected a JSON object body");
        }
        String generateConfigError = spec.validateRequest(body);
        if (generateConfigError != null) {
            return badRequest(generateConfigError);
        }
        JSONArray arr = body.get(spec.getRequestArrayField()) instanceof JSONArray value ? value : null;
        if (!spec.isSplittableBatch(body, arr)) {
            if (dryRun) {
                return preview("passthrough", List.of(body));
            }
            delegatedToPassthrough.set(true);
            return passthroughClient.forward(request, bytes);
        }
        String validationError = spec.validateForFanout(body);
        if (validationError != null) {
            return badRequest(validationError);
        }
        if (arr.isEmpty() && !dryRun) {
            return mergedResponse(ResponseMerger.merge(List.of(), spec, body));
        }

        TrafficPolicyConfig policy = loadBalanceConfig.getRouter().getGroupSelector();
        boolean preAssignmentAllowed = policy == null || (policy.getRules().isEmpty() && policy.getDefaultTargets().isEmpty());
        BatchChunkAssembler batch = new BatchChunkAssembler(body, spec, cfg.getSubBatchSpec());
        int chunkCount = batch.chunkCount();

        if (chunkCount > loadBalanceConfig.getRouter().getBatchScheduleMaxCount()) {
            return DispatcherResponses.error(413, "too_many_sub_batches",
                    "batch produces " + chunkCount + " sub-batches; maximum is "
                            + loadBalanceConfig.getRouter().getBatchScheduleMaxCount() + " (router.batchScheduleMaxCount)");
        }
        // Charge repeated envelopes before allocating targets or materializing chunks.
        if (batch.projectedBytes() > MAX_AGGREGATE_BYTES) {
            throw new AggregateRequestTooLargeException(MAX_AGGREGATE_BYTES);
        }

        if (dryRun) {
            return preview("split", batch.chunks(List.of()));
        }

        boolean assignBe = cfg.isPreAssignBe() && spec.isPreAssignable() && preAssignmentAllowed
                && loadBalanceConfig.getWorkerRegistry().getEngineType() == EngineType.LLM;
        return resolveTargets(chunkCount, assignBe)
                .publishOn(cpuScheduler)
                .flatMap(allocation -> {
                    if (!allocation.isSuccess()) {
                        int status = allocation.getCode() == StrategyErrorType.INVALID_REQUEST.getErrorCode()
                                ? 400 : 503;
                        return DispatcherResponses.error(status, "batch_schedule_failed",
                                allocation.getErrorMessage());
                    }
                    List<BatchScheduleTarget> targets = allocation.getServerStatus();
                    List<JSONObject> chunkBodies = batch.chunks(assignBe ? targets : List.of());
                    List<String> preAssignedFeUrls = targets.stream().map(BatchScheduleTarget::getFeUrl).toList();
                    return fanoutService.dispatchChunks(
                                    spec == BatchEndpointSpec.ROOT ? BatchEndpointSpec.BATCH_INFER.getPath() : spec.getPath(), chunkBodies,
                                    preAssignedFeUrls, spec,
                                    request.headers().asHttpHeaders(),
                                    request.uri().getRawQuery())
                            .publishOn(cpuScheduler)
                            .map(subs -> ResponseMerger.merge(subs, spec, body))
                            .flatMap(this::mergedResponse);
                });
    }

    /** Preview uses the outbound byte budget and stops before any target allocation. */
    private Mono<ServerResponse> preview(String mode, List<JSONObject> chunks) {
        byte[] body = BatchBodyParser.serialize(JSONObject.of("mode", mode, "chunk_count", chunks.size(), "chunks", chunks));
        if (body.length > MAX_AGGREGATE_BYTES) {
            throw new AggregateRequestTooLargeException(MAX_AGGREGATE_BYTES);
        }
        return DispatcherResponses.jsonBytes(200, body);
    }

    private Mono<ServerResponse> badRequest(String message) {
        return DispatcherResponses.error(400, "invalid_batch_request", message);
    }

    private Mono<ServerResponse> mergedResponse(ResponseMerger.MergedResponse merged) {
        return DispatcherResponses.jsonBytes(merged.status(), BatchBodyParser.serialize(merged.body()));
    }

    private Mono<BatchScheduleResponse> resolveTargets(
            int chunkCount, boolean assignBe) {
        BatchScheduleRequest request = new BatchScheduleRequest();
        request.setBatchCount(chunkCount);
        request.setAssignBe(assignBe);
        request.setAssignFe(true);
        return batchScheduleCoordinator.schedule(request);
    }
}
