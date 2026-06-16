package org.flexlb.httpserver;

import io.grpc.ManagedChannel;
import io.grpc.StatusRuntimeException;
import io.grpc.netty.NettyChannelBuilder;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.socket.nio.NioSocketChannel;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.FlexlbServiceGrpc;
import org.flexlb.config.ConfigService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

@Component
public class FlexlbGrpcForwarder {

    private final LBStatusConsistencyService lbStatusConsistencyService;
    private final ConfigService configService;
    private final EngineHealthReporter engineHealthReporter;
    private final EventLoopGroup eventLoopGroup;
    private final ConcurrentHashMap<String, ManagedChannel> channels = new ConcurrentHashMap<>();

    public FlexlbGrpcForwarder(LBStatusConsistencyService lbStatusConsistencyService,
                               ConfigService configService,
                               EngineHealthReporter engineHealthReporter,
                               @Qualifier("managedChannelEventLoopGroup") EventLoopGroup eventLoopGroup) {
        this.lbStatusConsistencyService = lbStatusConsistencyService;
        this.configService = configService;
        this.engineHealthReporter = engineHealthReporter;
        this.eventLoopGroup = eventLoopGroup;
    }

    public EngineRpcService.FlexlbScheduleResponsePB forwardToMaster(
            EngineRpcService.FlexlbScheduleRequestPB request) {
        String masterHostIpPort = lbStatusConsistencyService.getMasterHostIpPort();
        if (masterHostIpPort == null) {
            Logger.error("Master unreachable for gRPC forward, routing locally");
            engineHealthReporter.reportForwardToMasterResult("LOCAL", "MASTER_NULL");
            return null;
        }

        int grpcPort = resolveGrpcPort(masterHostIpPort);
        String ip = masterHostIpPort.split(":")[0];
        String channelKey = ip + ":" + grpcPort;

        try {
            ManagedChannel channel = channels.computeIfAbsent(channelKey, k -> createChannel(ip, grpcPort));
            FlexlbServiceGrpc.FlexlbServiceBlockingStub stub = FlexlbServiceGrpc.newBlockingStub(channel)
                    .withDeadlineAfter(configService.loadBalanceConfig().getPrefillLbTimeoutMs(), TimeUnit.MILLISECONDS);
            EngineRpcService.FlexlbScheduleResponsePB response = stub.schedule(request);
            engineHealthReporter.reportForwardToMasterResult(ip, String.valueOf(response.getCode()));
            return response;
        } catch (StatusRuntimeException e) {
            Logger.error("[Fallback] gRPC forward to master failed: {}, routing locally", e.getMessage());
            engineHealthReporter.reportForwardToMasterResult("LOCAL", "GRPC_FAILED");
            channels.remove(channelKey);
            return null;
        } catch (Exception e) {
            Logger.error("[Fallback] gRPC forward to master error, routing locally", e);
            engineHealthReporter.reportForwardToMasterResult("LOCAL", "CONNECT_FAILED");
            channels.remove(channelKey);
            return null;
        }
    }

    private int resolveGrpcPort(String masterHostIpPort) {
        // Always derive gRPC port from HTTP port using the same offset as FlexlbGrpcServer.
        String[] parts = masterHostIpPort.split(":");
        if (parts.length >= 2) {
            return Integer.parseInt(parts[1]) + FlexlbGrpcServer.FLEXLB_GRPC_PORT_OFFSET;
        }
        return 7001 + FlexlbGrpcServer.FLEXLB_GRPC_PORT_OFFSET;
    }

    private ManagedChannel createChannel(String ip, int port) {
        return NettyChannelBuilder.forAddress(ip, port)
                .channelType(NioSocketChannel.class)
                .eventLoopGroup(eventLoopGroup)
                .usePlaintext()
                .keepAliveTime(30, TimeUnit.SECONDS)
                .keepAliveTimeout(10, TimeUnit.SECONDS)
                .maxInboundMessageSize(16 * 1024 * 1024)
                .build();
    }

    @PreDestroy
    public void shutdown() {
        for (ManagedChannel channel : channels.values()) {
            channel.shutdownNow();
        }
        channels.clear();
    }
}
