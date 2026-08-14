package org.flexlb.httpserver;

import io.grpc.ManagedChannel;
import io.grpc.StatusRuntimeException;
import io.grpc.netty.NettyChannelBuilder;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.socket.nio.NioSocketChannel;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.schedule.grpc.FlexlbServiceGrpc;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol;
import org.flexlb.config.ConfigService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executor;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;

@Component
public class FlexlbGrpcForwarder {

    private final LBStatusConsistencyService lbStatusConsistencyService;
    private final ConfigService configService;
    private final EngineHealthReporter engineHealthReporter;
    private final EventLoopGroup eventLoopGroup;
    private final Executor executor;
    private final ConcurrentHashMap<String, ManagedChannel> channels = new ConcurrentHashMap<>();

    public FlexlbGrpcForwarder(LBStatusConsistencyService lbStatusConsistencyService,
                               ConfigService configService,
                               EngineHealthReporter engineHealthReporter,
                               @Qualifier("managedChannelEventLoopGroup") EventLoopGroup eventLoopGroup,
                               @Qualifier("forwarderChannelExecutor") Executor executor) {
        this.lbStatusConsistencyService = lbStatusConsistencyService;
        this.configService = configService;
        this.engineHealthReporter = engineHealthReporter;
        this.eventLoopGroup = eventLoopGroup;
        this.executor = executor;
    }

    public MasterForwardResult forwardToMaster(
            FlexlbScheduleProtocol.FlexlbScheduleRequestPB request) {
        String masterHostIpPort = lbStatusConsistencyService.getMasterHostIpPort();
        if (masterHostIpPort == null) {
            Logger.debug("Master unavailable for gRPC forward");
            engineHealthReporter.reportForwardToMasterResult("LOCAL", "MASTER_NULL");
            return MasterForwardResult.noMaster();
        }

        try {
            int grpcPort = resolveGrpcPort(masterHostIpPort);
            String ip = masterHostIpPort.split(":")[0];
            String channelKey = ip + ":" + grpcPort;
            ManagedChannel channel = channels.computeIfAbsent(
                    channelKey, k -> createChannel(ip, grpcPort));
            // The forward RPC inherits the inbound gRPC Context deadline. Do
            // not replace the request TTL with a load-balancer timeout.
            FlexlbServiceGrpc.FlexlbServiceBlockingStub stub =
                    FlexlbServiceGrpc.newBlockingStub(channel);
            FlexlbScheduleProtocol.FlexlbScheduleResponsePB response = stub.schedule(request);
            engineHealthReporter.reportForwardToMasterResult(ip, String.valueOf(response.getCode()));
            return MasterForwardResult.forwarded(response, masterHostIpPort);
        } catch (StatusRuntimeException e) {
            Logger.debug("gRPC forward to master failed: request_id={} master={} status={}",
                    request.getRequestId(), masterHostIpPort, e.getStatus().getCode());
            engineHealthReporter.reportForwardToMasterResult(ipOf(masterHostIpPort), "GRPC_FAILED");
            // The RPC may already have reached the Master. Report a terminal
            // result to the caller and never run a second local scheduler.
            // ManagedChannel reconnects itself; keep it cached until shutdown.
            return MasterForwardResult.failed(
                    e.getStatus().getCode().name(), masterHostIpPort);
        } catch (Exception e) {
            Logger.error("gRPC forward to master error: request_id={} master={}",
                    request.getRequestId(), masterHostIpPort, e);
            engineHealthReporter.reportForwardToMasterResult(
                    ipOf(masterHostIpPort), "CONNECT_FAILED");
            return MasterForwardResult.failed(
                    e.getClass().getSimpleName(), masterHostIpPort);
        }
    }

    public record MasterForwardResult(
            FlexlbScheduleProtocol.FlexlbScheduleResponsePB response,
            boolean masterFound,
            String failure,
            String masterHost) {

        static MasterForwardResult forwarded(
                FlexlbScheduleProtocol.FlexlbScheduleResponsePB response,
                String masterHost) {
            return new MasterForwardResult(response, true, "", masterHost);
        }

        static MasterForwardResult noMaster() {
            return new MasterForwardResult(null, false, "MASTER_NULL", "");
        }

        static MasterForwardResult failed(String failure, String masterHost) {
            return new MasterForwardResult(null, true, failure, masterHost);
        }
    }

    public FlexlbScheduleProtocol.GetRequestStateResponsePB forwardGetRequestStateToMaster(
            FlexlbScheduleProtocol.GetRequestStateRequestPB request) {
        return invokeMaster("state query", request.getRequestId(),
                stub -> stub.getRequestState(request));
    }

    private <T> T invokeMaster(String operation,
                               long requestId,
                               Function<FlexlbServiceGrpc.FlexlbServiceBlockingStub, T> rpc) {
        FlexlbServiceGrpc.FlexlbServiceBlockingStub stub = masterStub();
        if (stub == null) {
            return null;
        }
        try {
            return rpc.apply(stub);
        } catch (RuntimeException e) {
            Logger.debug("Failed to forward FlexLB {} to master, request_id={}",
                    operation, requestId, e);
            return null;
        }
    }

    private FlexlbServiceGrpc.FlexlbServiceBlockingStub masterStub() {
        String masterHostIpPort = lbStatusConsistencyService.getMasterHostIpPort();
        if (masterHostIpPort == null) {
            return null;
        }
        int grpcPort = resolveGrpcPort(masterHostIpPort);
        String ip = masterHostIpPort.split(":")[0];
        String channelKey = ip + ":" + grpcPort;
        ManagedChannel channel = channels.computeIfAbsent(channelKey, k -> createChannel(ip, grpcPort));
        return FlexlbServiceGrpc.newBlockingStub(channel)
                .withDeadlineAfter(configService.loadBalanceConfig().getPrefillLbTimeoutMs(), TimeUnit.MILLISECONDS);
    }

    private int resolveGrpcPort(String masterHostIpPort) {
        // Always derive gRPC port from HTTP port using the same offset as FlexlbGrpcServer.
        String[] parts = masterHostIpPort.split(":");
        if (parts.length >= 2) {
            return Integer.parseInt(parts[1]) + FlexlbGrpcServer.FLEXLB_GRPC_PORT_OFFSET;
        }
        return 7001 + FlexlbGrpcServer.FLEXLB_GRPC_PORT_OFFSET;
    }

    private static String ipOf(String hostIpPort) {
        return hostIpPort.split(":", 2)[0];
    }

    private ManagedChannel createChannel(String ip, int port) {
        return NettyChannelBuilder.forAddress(ip, port)
                .channelType(NioSocketChannel.class)
                .eventLoopGroup(eventLoopGroup)
                .executor(executor)
                .usePlaintext()
                .disableRetry()
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
