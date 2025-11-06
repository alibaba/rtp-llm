package org.flexlb.engine.grpc.monitor;

import org.flexlb.enums.FlexMetricType;
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

    @PostConstruct
    public void init() {
        this.monitor.register(GRPC_CHANNEL_POOL_SIZE, FlexMetricType.GAUGE);
    }

    /**
     * Report the current size of the gRPC channel pool
     *
     * @param channelPoolSize current number of channels in the pool
     */
    public void reportChannelPoolSize(int channelPoolSize) {
        monitor.report(GRPC_CHANNEL_POOL_SIZE, FlexMetricTags.of("type", "grpc"), channelPoolSize);
    }
}