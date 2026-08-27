package org.flexlb.service.optimizer;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.config.OptimizerRuntimeConfig;
import org.flexlb.constant.MetricConstant;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.optimizer.CommonResponseHeader;
import org.flexlb.dao.optimizer.OptimizerErrorCode;
import org.flexlb.dao.optimizer.OptimizerTraceQueryRequest;
import org.flexlb.dao.optimizer.OptimizerTraceQueryResponse;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.dao.route.OptimizerConfig;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.engine.grpc.client.KvcmWorkerMetadataResolver;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.transport.GeneralHttpNettyService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import java.net.URI;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

@Slf4j
@Component
public class OptimizerClient {

    private static final String PATH_TRACE_QUERY = "/traceQuery";
    private final GeneralHttpNettyService httpService;
    private final OptimizerAddressResolver addressResolver;
    private final KvcmWorkerMetadataResolver workerMetadataResolver;
    private final FlexMonitor monitor;
    private final String basePath;
    private final boolean enabled;
    private final AtomicBoolean shutdown = new AtomicBoolean(false);

    @Autowired
    public OptimizerClient(GeneralHttpNettyService httpService,
                           ServiceDiscovery serviceDiscovery,
                           ModelMetaConfig modelMetaConfig,
                           ConfigService configService,
                           KvcmWorkerMetadataResolver workerMetadataResolver,
                           FlexMonitor monitor) {
        this.httpService = httpService;
        this.workerMetadataResolver = workerMetadataResolver;
        this.monitor = monitor;

        OptimizerRuntimeConfig runtimeConfig =
                configService.loadBalanceConfig().getOptimizer();
        OptimizerConfig optimizerConfig = resolveOptimizerConfig(modelMetaConfig);
        this.enabled = runtimeConfig.isEnabled();
        if (!enabled) {
            this.addressResolver = null;
            this.basePath = "";
            return;
        }
        if (optimizerConfig == null) {
            throw new IllegalStateException(
                    "FLEXLB_CONFIG optimizer.enabled=true requires "
                            + "MODEL_SERVICE_CONFIG optimizer topology");
        }

        Endpoint endpoint = optimizerConfig.toEndpoint();
        this.addressResolver =
                new OptimizerAddressResolver(
                        serviceDiscovery,
                        endpoint,
                        optimizerConfig.getPort(),
                        runtimeConfig.getDiscoveryPollIntervalMs());
        this.basePath = stripTrailingSlash(optimizerConfig.getPath());
        log.info("Optimizer trace query enabled: address={}", endpoint.getAddress());
    }

    OptimizerClient(
            GeneralHttpNettyService httpService,
            OptimizerAddressResolver addressResolver,
            KvcmWorkerMetadataResolver workerMetadataResolver,
            String basePath,
            FlexMonitor monitor) {
        this.httpService = httpService;
        this.addressResolver = addressResolver;
        this.workerMetadataResolver = workerMetadataResolver;
        this.basePath = stripTrailingSlash(basePath);
        this.monitor = monitor;
        this.enabled = true;
    }

    @PostConstruct
    public void init() {
        monitor.register(MetricConstant.OPTIMIZER_TRACE_QUERY_SKIPPED_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(MetricConstant.OPTIMIZER_TRACE_QUERY_FAILED_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        if (!enabled) {
            return;
        }
        try {
            addressResolver.start();
        } catch (Exception e) {
            log.warn("Optimizer discovery resolver failed to start", e);
        }
    }

    public void traceQuery(Request request, ServerStatus selectedWorker) {
        if (!enabled) {
            return;
        }
        if (shutdown.get()) {
            reportSkipped("shutdown");
            return;
        }
        if (request == null || request.getBlockCacheKeys() == null || request.getBlockCacheKeys().isEmpty()) {
            reportSkipped("empty_block_keys");
            return;
        }

        try {
            String instanceId = resolveInstanceId(request, selectedWorker);
            if (StringUtils.isBlank(instanceId)) {
                reportSkipped("instance_id_unavailable");
                return;
            }
            List<String> addresses = addressResolver.getAddresses();
            if (addresses == null || addresses.isEmpty()) {
                reportSkipped("no_available_address");
                return;
            }

            OptimizerTraceQueryRequest traceQueryRequest = new OptimizerTraceQueryRequest();
            traceQueryRequest.setTraceId(String.valueOf(request.getRequestId()));
            traceQueryRequest.setInstanceId(instanceId);
            traceQueryRequest.setBlockKeys(request.getBlockCacheKeys());
            traceQueryRequest.setTokenIds(List.of());
            traceQueryRequest.setInputTokenLen(request.getSeqLen());

            URI uri = URI.create("http://" + addresses.getFirst());
            httpService.request(traceQueryRequest, uri, basePath + PATH_TRACE_QUERY,
                            OptimizerTraceQueryResponse.class)
                    .subscribe(
                            this::handleTraceQueryResponse,
                            error -> reportFailed("http_error", request.getRequestId(), uri, error));
        } catch (Exception e) {
            reportFailed("dispatch_error", request == null ? null : request.getRequestId(), null, e);
        }
    }

    void handleTraceQueryResponse(OptimizerTraceQueryResponse response) {
        OptimizerErrorCode statusCode = extractStatusCode(response);
        if (statusCode == OptimizerErrorCode.OK) {
            return;
        }
        reportFailed("status_" + (statusCode == null ? "MISSING" : statusCode.name()));
    }

    private void reportSkipped(String reason) {
        monitor.report(MetricConstant.OPTIMIZER_TRACE_QUERY_SKIPPED_QPS,
                FlexMetricTags.of("reason", reason), 1.0);
        log.warn("Optimizer trace query skipped: reason={}", reason);
    }

    private void reportFailed(String reason) {
        monitor.report(MetricConstant.OPTIMIZER_TRACE_QUERY_FAILED_QPS,
                FlexMetricTags.of("reason", reason), 1.0);
        log.warn("Optimizer trace query failed: reason={}", reason);
    }

    private void reportFailed(String reason, Long requestId, URI uri, Throwable error) {
        monitor.report(MetricConstant.OPTIMIZER_TRACE_QUERY_FAILED_QPS,
                FlexMetricTags.of("reason", reason), 1.0);
        log.warn("Optimizer trace query failed: reason={}, requestId={}, uri={}",
                reason, requestId, uri, error);
    }

    private String resolveInstanceId(Request request, ServerStatus selectedWorker) {
        if (selectedWorker == null || selectedWorker.getRole() == null || request.getBlockSize() <= 0) {
            return null;
        }
        return workerMetadataResolver.resolveNamespace(
                selectedWorker.getRole(), selectedWorker.getGroup(), request.getBlockSize());
    }

    @PreDestroy
    public void shutdown() {
        if (!shutdown.compareAndSet(false, true)) {
            return;
        }
        if (addressResolver != null) {
            addressResolver.shutdown();
        }
    }

    private static OptimizerConfig resolveOptimizerConfig(ModelMetaConfig modelMetaConfig) {
        for (ServiceRoute route : modelMetaConfig.getServiceRoutes()) {
            if (route != null && route.getOptimizer() != null) {
                return route.getOptimizer();
            }
        }
        return null;
    }

    private static String stripTrailingSlash(String path) {
        return StringUtils.stripEnd(StringUtils.defaultString(path), "/");
    }

    private static OptimizerErrorCode extractStatusCode(OptimizerTraceQueryResponse response) {
        if (response == null) {
            return null;
        }
        CommonResponseHeader header = response.getHeader();
        if (header == null || header.getStatus() == null) {
            return null;
        }
        return header.getStatus().getCode();
    }
}
