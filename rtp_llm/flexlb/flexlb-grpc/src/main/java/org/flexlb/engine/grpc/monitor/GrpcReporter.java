package org.flexlb.engine.grpc.monitor;

import org.flexlb.engine.grpc.GrpcCallTrace;
import org.flexlb.engine.grpc.GrpcFailure;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.metric.SafeFlexMonitor;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;

import static org.flexlb.constant.MetricConstant.GRPC_CALL_COUNT;
import static org.flexlb.constant.MetricConstant.GRPC_CALL_DURATION;
import static org.flexlb.constant.MetricConstant.GRPC_CALL_FAILURE;
import static org.flexlb.constant.MetricConstant.GRPC_CHANNEL_POOL_SIZE;
import static org.flexlb.constant.MetricConstant.GRPC_CONNECTION_DURATION;
import static org.flexlb.constant.MetricConstant.GRPC_DELIVERY_DELAY;

/**
 * Reporter for gRPC channel pool metrics.
 * Metric registration and reporting are best-effort and do not interrupt RPC processing.
 *
 * Note: All gRPC metrics use "engineIp" as the IP tag name (previously "ip"),
 * consistent with other FlexLB reporters (EngineHealthReporter, DeliveryMetricsReporter,
 * CacheMetricsReporter, etc.). If any Grafana dashboard queries reference the old "ip"
 * tag for these gRPC metrics, they must be updated to use "engineIp" instead.
 */
@Component
public class GrpcReporter {

    private final FlexMonitor monitor;

    public GrpcReporter(FlexMonitor monitor) {
        this.monitor = SafeFlexMonitor.wrap(monitor);
    }

    @PostConstruct
    public void init() {
        SafeFlexMonitor.run(
                GRPC_CALL_FAILURE,
                () ->
                        monitor.register(
                                GRPC_CALL_FAILURE, FlexMetricType.QPS, FlexPriorityType.PRECISE));
        SafeFlexMonitor.run(
                GRPC_DELIVERY_DELAY,
                () ->
                        monitor.register(
                                GRPC_DELIVERY_DELAY,
                                FlexMetricType.GAUGE,
                                FlexPriorityType.PRECISE));
        monitor.register(GRPC_CHANNEL_POOL_SIZE, FlexMetricType.GAUGE);
        monitor.register(GRPC_CALL_DURATION, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(GRPC_CALL_COUNT, FlexMetricType.QPS);
        monitor.register(GRPC_CONNECTION_DURATION, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
    }

    /**
     * Report the current size of the gRPC channel pool
     *
     * @param channelPoolSize current number of channels in the pool
     */
    public void reportChannelPoolSize(int channelPoolSize) {
        monitor.report(GRPC_CHANNEL_POOL_SIZE, FlexMetricTags.of("type", "grpc"), channelPoolSize);
    }

    /**
     * Report gRPC call metrics
     *
     * @param serviceType Service type
     * @param duration Call duration in milliseconds
     * @param isRetry Whether this is a retry call
     */
    public void reportCallMetrics(String serviceType, long duration, boolean isRetry) {
        FlexMetricTags tags =
                FlexMetricTags.of("service", serviceType, "retry", String.valueOf(isRetry));

        // Report call duration
        monitor.report(GRPC_CALL_DURATION, tags, duration);

        // Report call count
        monitor.report(GRPC_CALL_COUNT, tags, 1);
    }

    /**
     * Report gRPC connection duration
     *
     * @param serviceType Service type
     * @param connectionDuration Connection duration in microseconds
     */
    public void reportConnectionDuration(String serviceType, long connectionDuration) {
        FlexMetricTags tags = FlexMetricTags.of("service", serviceType);

        // Report connection duration
        monitor.report(GRPC_CONNECTION_DURATION, tags, connectionDuration);
    }

    /** Reports one logical RPC only after its final failure, including any retry. */
    public void reportCallFailure(
            String serviceType,
            boolean retry,
            String phase,
            GrpcFailure.Analysis failure,
            GrpcCallTrace.Snapshot transport) {
        FlexMetricTags tags =
                FlexMetricTags.of(
                        "service", serviceType,
                        "retry", String.valueOf(retry),
                        "phase", phase,
                        "reason", failure.kind().name(),
                        "origin", transport.failureOrigin(failure),
                        "grpc_code", failure.grpcCode().name(),
                        "transport_stage", transport.stage().name(),
                        "transport_cause",
                                transport.transportCauseKind() == null
                                        ? "NONE"
                                        : transport.transportCauseKind().name(),
                        "transport_event",
                                "NONE".equals(transport.transportEvent())
                                                && failure.kind()
                                                        == GrpcFailure.Kind.CONNECT_REFUSED
                                        ? "CONNECT_REFUSED"
                                        : transport.transportEvent());
        monitor.report(GRPC_CALL_FAILURE, tags, 1.0);
    }

    /** Local transport-to-Future delivery time, including successful calls. Unknown samples are omitted. */
    public void reportDeliveryDelay(
            String serviceType, boolean retry, GrpcCallTrace.Snapshot transport) {
        if (transport.callbackDelayMs() >= 0) {
            monitor.report(
                    GRPC_DELIVERY_DELAY,
                    FlexMetricTags.of(
                            "service",
                            serviceType,
                            "retry",
                            String.valueOf(retry),
                            "stage",
                            "STREAM_CLOSED_TO_CALLBACK"),
                    transport.callbackDelayMs());
        }
        if (transport.responseDeliveryMs() >= 0) {
            monitor.report(
                    GRPC_DELIVERY_DELAY,
                    FlexMetricTags.of(
                            "service",
                            serviceType,
                            "retry",
                            String.valueOf(retry),
                            "stage",
                            "TRAILERS_TO_CALLBACK"),
                    transport.responseDeliveryMs());
        }
    }
}
