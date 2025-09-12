package org.flexlb.engine.grpc.monitor;

import javax.annotation.PostConstruct;

import com.taobao.kmonitor.ImmutableMetricTags;
import com.taobao.kmonitor.KMonitor;
import com.taobao.kmonitor.MetricType;
import org.springframework.stereotype.Component;

import static org.flexlb.constant.MetricConstant.GRPC_CHANNEL_POOL_SIZE;

/**
 * Reporter for gRPC channel pool metrics
 */
@Component
public class GrpcReporter {

    private final KMonitor monitor;

    public GrpcReporter(KMonitor monitor) {
        this.monitor = monitor;
    }

    @PostConstruct
    public void init() {
        this.monitor.register(GRPC_CHANNEL_POOL_SIZE, MetricType.GAUGE);
    }

    /**
     * Report the current size of the gRPC channel pool
     *
     * @param channelPoolSize current number of channels in the pool
     */
    public void reportChannelPoolSize(int channelPoolSize) {
        monitor.report(GRPC_CHANNEL_POOL_SIZE, new ImmutableMetricTags("type", "grpc"), channelPoolSize);
    }
}