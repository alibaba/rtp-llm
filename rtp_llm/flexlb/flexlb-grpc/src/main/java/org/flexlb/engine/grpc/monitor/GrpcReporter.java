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

    // gRPC 调用相关指标常量
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
     * 上报 gRPC 调用指标
     * 
     * @param ip 目标 IP
     * @param serviceType 服务类型
     * @param duration 调用耗时（毫秒）
     * @param responseSize 响应体大小（字节）
     * @param isRetry 是否为重试调用
     */
    public void reportCallMetrics(String ip, String serviceType, long duration, int responseSize, boolean isRetry) {
        FlexMetricTags tags = FlexMetricTags.of(
            "ip", ip,
            "service", serviceType,
            "retry", String.valueOf(isRetry)
        );
        
        // 上报调用耗时
        monitor.report(GRPC_CALL_DURATION, tags, duration);
        
        // 上报响应体大小
        monitor.report(GRPC_RESPONSE_SIZE, tags, responseSize);
        
        // 上报调用次数
        monitor.report(GRPC_CALL_COUNT, tags, 1);
    }
    
    /**
     * 上报 gRPC 连接持续时间
     * 
     * @param ip 目标 IP
     * @param serviceType 服务类型
     * @param connectionDuration 连接持续时间（微秒）
     */
    public void reportConnectionDuration(String ip, String serviceType, long connectionDuration) {
        FlexMetricTags tags = FlexMetricTags.of(
            "ip", ip,
            "service", serviceType
        );
        
        // 上报连接持续时间
        monitor.report(GRPC_CONNECTION_DURATION, tags, connectionDuration);
    }
}