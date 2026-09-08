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

@Component
public class FlexlbGrpcForwarder {

    static final int MAX_FORWARD_HOPS = 1;

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
        ForwardGuard guard = applyForwardGuard(
                request.getRequestId(), request.getForwardHop(),
                ForwardOperation.SCHEDULE);
        if (guard.blocked()) {
            return MasterForwardResult.failed(
                    guard.blockReason().failureCode(),
                    nullToEmpty(guard.masterHostIpPort()));
        }

        String masterHostIpPort = guard.masterHostIpPort();
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
            FlexlbScheduleProtocol.FlexlbScheduleRequestPB forwardedRequest =
                    request.toBuilder().setForwardHop(guard.nextHop()).build();
            FlexlbScheduleProtocol.FlexlbScheduleResponsePB response =
                    stub.schedule(forwardedRequest);
            engineHealthReporter.reportForwardToMasterResult(ip, String.valueOf(response.getCode()));
            return MasterForwardResult.forwarded(response, masterHostIpPort);
        } catch (StatusRuntimeException e) {
            Logger.warn(
                    "event=flexlb_forward_failed request_id={} forward_hop={} master={} "
                            + "local_ip={} status={}",
                    request.getRequestId(), guard.nextHop(), masterHostIpPort,
                    guard.localIp(), e.getStatus().getCode());
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
        ForwardGuard guard = applyForwardGuard(
                request.getRequestId(), request.getForwardHop(),
                ForwardOperation.STATE_QUERY);
        if (guard.blocked()) {
            return null;
        }
        String masterHostIpPort = guard.masterHostIpPort();
        FlexlbServiceGrpc.FlexlbServiceBlockingStub stub = masterStub(masterHostIpPort);
        if (stub == null) {
            return null;
        }
        FlexlbScheduleProtocol.GetRequestStateRequestPB forwardedRequest =
                request.toBuilder().setForwardHop(guard.nextHop()).build();
        try {
            return stub.getRequestState(forwardedRequest);
        } catch (RuntimeException e) {
            Logger.debug("Failed to forward FlexLB state query to master, request_id={}",
                    request.getRequestId(), e);
            return null;
        }
    }

    private FlexlbServiceGrpc.FlexlbServiceBlockingStub masterStub(
            String masterHostIpPort) {
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

    private ForwardGuard applyForwardGuard(
            long requestId,
            int encodedHop,
            ForwardOperation operation) {
        long incomingHop = Integer.toUnsignedLong(encodedHop);
        String masterHostIpPort = lbStatusConsistencyService.getMasterHostIpPort();
        String localIp = lbStatusConsistencyService.getLocalHostIp();
        ForwardBlockReason blockReason = incomingHop >= MAX_FORWARD_HOPS
                ? ForwardBlockReason.HOP_LIMIT
                : sameHost(localIp, masterHostIpPort)
                        ? ForwardBlockReason.SELF_TARGET
                        : null;
        ForwardGuard guard = new ForwardGuard(
                incomingHop, masterHostIpPort, localIp, blockReason);
        if (!guard.blocked()) {
            return guard;
        }

        Logger.warn(
                "event=flexlb_forward_blocked request_id={} operation={} reason={} "
                        + "forward_hop={} local_ip={} cached_master={} is_master={}",
                requestId, operation.logValue(), blockReason.name(), incomingHop,
                localIp, masterHostIpPort, lbStatusConsistencyService.isMaster());
        engineHealthReporter.reportForwardToMasterResult(
                ipOfOrLocal(masterHostIpPort), blockReason.name());
        return guard;
    }

    private enum ForwardOperation {
        SCHEDULE("schedule"),
        STATE_QUERY("state_query");

        private final String logValue;

        ForwardOperation(String logValue) {
            this.logValue = logValue;
        }

        String logValue() {
            return logValue;
        }
    }

    private enum ForwardBlockReason {
        HOP_LIMIT("FORWARD_HOP_LIMIT"),
        SELF_TARGET("SELF_FORWARD_BLOCKED");

        private final String failureCode;

        ForwardBlockReason(String failureCode) {
            this.failureCode = failureCode;
        }

        String failureCode() {
            return failureCode;
        }
    }

    private record ForwardGuard(
            long incomingHop,
            String masterHostIpPort,
            String localIp,
            ForwardBlockReason blockReason) {

        boolean blocked() {
            return blockReason != null;
        }

        int nextHop() {
            return Math.toIntExact(incomingHop + 1);
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

    private static String ipOf(String hostIpPort) {
        return hostIpPort.split(":", 2)[0];
    }

    private static boolean sameHost(String localIp, String hostIpPort) {
        return localIp != null && hostIpPort != null
                && localIp.equals(ipOf(hostIpPort));
    }

    private static String ipOfOrLocal(String hostIpPort) {
        return hostIpPort == null || hostIpPort.isBlank()
                ? "LOCAL"
                : ipOf(hostIpPort);
    }

    private static String nullToEmpty(String value) {
        return value == null ? "" : value;
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
