package org.flexlb.engine.grpc.monitor;

import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;

import static org.flexlb.constant.MetricConstant.GRPC_CHANNEL_POOL_SIZE;

/**
 * Reporter for gRPC channel pool metrics
 */
@Component
public class GrpcReporter {

    private final FlexMonitor monitor;

    public GrpcReporter(FlexMonitor monitor) {
        this.monitor = monitor;
    }

    // gRPC call related metric constants
    private static final String GRPC_CALL_DURATION = "grpc.call.duration";
    private static final String GRPC_RESPONSE_SIZE = "grpc.response.size";
    private static final String GRPC_CALL_COUNT = "grpc.call.count";
    private static final String GRPC_CONNECTION_DURATION = "grpc.connection.duration";

    @PostConstruct
    public void init() {
        this.monitor.register(GRPC_CHANNEL_POOL_SIZE, FlexMetricType.GAUGE);
        this.monitor.register(GRPC_CALL_DURATION, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(GRPC_RESPONSE_SIZE, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(GRPC_CALL_COUNT, FlexMetricType.QPS);
        this.monitor.register(GRPC_CONNECTION_DURATION, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
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
     * @param ip Target IP address
     * @param serviceType Service type
     * @param duration Call duration in milliseconds
     * @param responseSize Response body size in bytes
     * @param isRetry Whether this is a retry call
     */
    public void reportCallMetrics(String ip, String serviceType, long duration, int responseSize, boolean isRetry) {
        FlexMetricTags tags = FlexMetricTags.of(
            "ip", ip,
            "service", serviceType,
            "retry", String.valueOf(isRetry)
        );

        // Report call duration
        monitor.report(GRPC_CALL_DURATION, tags, duration);

        // Report response body size
        monitor.report(GRPC_RESPONSE_SIZE, tags, responseSize);

        // Report call count
        monitor.report(GRPC_CALL_COUNT, tags, 1);
    }
    
    /**
     * Report gRPC connection duration
     *
     * @param ip Target IP address
     * @param serviceType Service type
     * @param connectionDuration Connection duration in microseconds
     */
    public void reportConnectionDuration(String ip, String serviceType, long connectionDuration) {
        FlexMetricTags tags = FlexMetricTags.of(
            "ip", ip,
            "service", serviceType
        );

        // Report connection duration
        monitor.report(GRPC_CONNECTION_DURATION, tags, connectionDuration);
    }
}