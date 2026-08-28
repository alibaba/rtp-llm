package org.flexlb.httpserver;

import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import io.grpc.ManagedChannel;
import io.grpc.Status;
import io.grpc.StatusException;
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
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

@Component
public class FlexlbGrpcForwarder {

    static final int MAX_FORWARD_HOPS = 1;

    private final LBStatusConsistencyService lbStatusConsistencyService;
    private final ConfigService configService;
    private final EngineHealthReporter engineHealthReporter;
    private final EventLoopGroup eventLoopGroup;
    private final Executor executor;
    private final ConcurrentHashMap<String, ManagedChannel> channels = new ConcurrentHashMap<>();
    private final AtomicBoolean shutdown = new AtomicBoolean(false);

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

    public CompletionStage<MasterForwardResult> forwardScheduleToMaster(
            FlexlbScheduleProtocol.FlexlbScheduleRequestPB request) {
        ForwardGuard guard = applyForwardGuard(
                request.getRequestId(), request.getForwardHop(),
                ForwardOperation.SCHEDULE);
        if (guard.blocked()) {
            return CompletableFuture.completedFuture(MasterForwardResult.failed(
                    guard.blockReason().failureCode(),
                    nullToEmpty(guard.masterHostIpPort())));
        }

        String masterHostIpPort = guard.masterHostIpPort();
        if (masterHostIpPort == null) {
            Logger.debug("Master unavailable for gRPC forward");
            reportForwardResult("LOCAL", "MASTER_NULL");
            return CompletableFuture.completedFuture(MasterForwardResult.noMaster());
        }

        String masterIp = ipOf(masterHostIpPort);
        FlexlbScheduleProtocol.FlexlbScheduleRequestPB forwardedRequest =
                request.toBuilder().setForwardHop(guard.nextHop()).build();
        ListenableFuture<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> rpcFuture;
        try {
            // The forward RPC inherits the inbound gRPC Context deadline. Do
            // not replace the request TTL with a load-balancer timeout.
            rpcFuture = FlexlbServiceGrpc.newFutureStub(masterChannel(masterHostIpPort))
                    .schedule(forwardedRequest);
        } catch (RuntimeException error) {
            return CompletableFuture.completedFuture(forwardFailure(
                    request.getRequestId(), guard, error));
        }

        CompletableFuture<MasterForwardResult> result = new CompletableFuture<>();
        try {
            Futures.addCallback(rpcFuture,
                    new FutureCallback<>() {
                        @Override
                        public void onSuccess(
                                FlexlbScheduleProtocol.FlexlbScheduleResponsePB response) {
                            if (response == null) {
                                result.complete(MasterForwardResult.failed(
                                        "MISSING_RESPONSE", masterHostIpPort));
                                return;
                            }
                            reportForwardResult(masterIp, String.valueOf(response.getCode()));
                            result.complete(MasterForwardResult.forwarded(
                                    response, masterHostIpPort));
                        }

                        @Override
                        public void onFailure(Throwable error) {
                            result.complete(forwardFailure(
                                    request.getRequestId(), guard, error));
                        }
                    },
                    Runnable::run);
        } catch (RuntimeException callbackRegistrationError) {
            rpcFuture.cancel(true);
            result.complete(forwardFailure(
                    request.getRequestId(), guard, callbackRegistrationError));
        }

        // gRPC already propagates the inbound Context deadline/cancellation to
        // the client call. Also propagate an explicit cancellation by a direct
        // caller of this method; this does not allocate or schedule a worker.
        result.whenComplete((ignored, error) -> {
            if (result.isCancelled()) {
                rpcFuture.cancel(true);
            }
        });
        return result;
    }

    /**
     * Forward cancellation to the lifecycle-owning Master without blocking
     * the follower's gRPC request thread.
     *
     * <p>The request carries the same single-hop fence as Schedule, so a stale
     * election view cannot relay cancellation through a second follower. The
     * forwarded result remains authoritative: callers must not retry the same
     * cancellation locally after an RPC was attempted.</p>
     */
    public CompletionStage<CancelForwardResult> forwardCancelToMaster(
            FlexlbScheduleProtocol.FlexlbCancelRequestPB request) {
        ForwardGuard guard = applyForwardGuard(
                request.getRequestId(), request.getForwardHop(), ForwardOperation.CANCEL);
        if (guard.blocked()) {
            return CompletableFuture.completedFuture(CancelForwardResult.failed(
                    guard.blockReason().failureCode(),
                    nullToEmpty(guard.masterHostIpPort())));
        }

        String masterHostIpPort = guard.masterHostIpPort();
        if (masterHostIpPort == null) {
            Logger.debug("Master unavailable for cancellation forward");
            reportForwardResult("LOCAL", "MASTER_NULL");
            return CompletableFuture.completedFuture(CancelForwardResult.noMaster());
        }

        String masterIp = ipOf(masterHostIpPort);
        FlexlbScheduleProtocol.FlexlbCancelRequestPB forwardedRequest =
                request.toBuilder().setForwardHop(guard.nextHop()).build();
        ListenableFuture<FlexlbScheduleProtocol.FlexlbCancelResponsePB> rpcFuture;
        try {
            // As with Schedule, the future stub inherits the inbound gRPC
            // Context deadline and cancellation.
            rpcFuture = FlexlbServiceGrpc.newFutureStub(masterChannel(masterHostIpPort))
                    .cancel(forwardedRequest);
        } catch (RuntimeException error) {
            return CompletableFuture.completedFuture(cancelForwardFailure(
                    request.getRequestId(), guard, error));
        }

        CompletableFuture<CancelForwardResult> result = new CompletableFuture<>();
        try {
            Futures.addCallback(rpcFuture,
                    new FutureCallback<>() {
                        @Override
                        public void onSuccess(
                                FlexlbScheduleProtocol.FlexlbCancelResponsePB response) {
                            if (response == null) {
                                result.complete(CancelForwardResult.failed(
                                        "MISSING_RESPONSE", masterHostIpPort));
                                return;
                            }
                            reportForwardResult(masterIp,
                                    response.getFound()
                                            ? "CANCEL_FOUND"
                                            : "CANCEL_NOT_FOUND");
                            result.complete(CancelForwardResult.forwarded(
                                    response, masterHostIpPort));
                        }

                        @Override
                        public void onFailure(Throwable error) {
                            result.complete(cancelForwardFailure(
                                    request.getRequestId(), guard, error));
                        }
                    },
                    Runnable::run);
        } catch (RuntimeException callbackRegistrationError) {
            rpcFuture.cancel(true);
            result.complete(cancelForwardFailure(
                    request.getRequestId(), guard, callbackRegistrationError));
        }

        result.whenComplete((ignored, error) -> {
            if (result.isCancelled()) {
                rpcFuture.cancel(true);
            }
        });
        return result;
    }

    private MasterForwardResult forwardFailure(
            long requestId,
            ForwardGuard guard,
            Throwable error) {
        return MasterForwardResult.failed(
                recordForwardFailure(requestId, guard, error),
                nullToEmpty(guard.masterHostIpPort()));
    }

    private CancelForwardResult cancelForwardFailure(
            long requestId,
            ForwardGuard guard,
            Throwable error) {
        return CancelForwardResult.failed(
                recordForwardFailure(requestId, guard, error),
                nullToEmpty(guard.masterHostIpPort()));
    }

    private String recordForwardFailure(
            long requestId,
            ForwardGuard guard,
            Throwable error) {
        Status status = Status.fromThrowable(error);
        boolean grpcFailure = error instanceof StatusException
                || error instanceof StatusRuntimeException
                || status.getCode() != Status.Code.UNKNOWN;
        String failure = grpcFailure
                ? status.getCode().name()
                : error.getClass().getSimpleName();
        String masterHost = nullToEmpty(guard.masterHostIpPort());
        if (grpcFailure) {
            Logger.warn(
                    "event=flexlb_forward_failed request_id={} forward_hop={} master={} "
                            + "local_ip={} status={}",
                    requestId, guard.nextHop(), masterHost,
                    guard.localIp(), status.getCode());
            reportForwardResult(ipOfOrLocal(masterHost), "GRPC_FAILED");
        } else {
            Logger.error("gRPC forward to master error: request_id={} master={}",
                    requestId, masterHost, error);
            reportForwardResult(ipOfOrLocal(masterHost), "CONNECT_FAILED");
        }
        // The RPC may already have reached the Master. Keep this terminal for
        // the caller and let the ManagedChannel reconnect itself.
        return failure;
    }

    private void reportForwardResult(String target, String result) {
        try {
            engineHealthReporter.reportForwardToMasterResult(target, result);
        } catch (RuntimeException error) {
            // Observability must never suppress or duplicate an RPC result.
            Logger.warn("Failed to report forward result: target={} result={}",
                    target, result, error);
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

    public record CancelForwardResult(
            FlexlbScheduleProtocol.FlexlbCancelResponsePB response,
            boolean masterFound,
            String failure,
            String masterHost) {

        static CancelForwardResult forwarded(
                FlexlbScheduleProtocol.FlexlbCancelResponsePB response,
                String masterHost) {
            return new CancelForwardResult(response, true, "", masterHost);
        }

        static CancelForwardResult noMaster() {
            return new CancelForwardResult(null, false, "MASTER_NULL", "");
        }

        static CancelForwardResult failed(String failure, String masterHost) {
            return new CancelForwardResult(null, true, failure, masterHost);
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
        try {
            return FlexlbServiceGrpc.newBlockingStub(masterChannel(masterHostIpPort))
                    .withDeadlineAfter(
                            configService.loadBalanceConfig().getInternalRuntime()
                                    .getMasterForwardRpcTimeoutMs(),
                            TimeUnit.MILLISECONDS);
        } catch (RuntimeException error) {
            Logger.debug("Failed to create FlexLB master stub for {}",
                    masterHostIpPort, error);
            return null;
        }
    }

    private ManagedChannel masterChannel(String masterHostIpPort) {
        if (shutdown.get()) {
            throw Status.UNAVAILABLE
                    .withDescription("FlexLB gRPC forwarder is shutting down")
                    .asRuntimeException();
        }
        int grpcPort = resolveGrpcPort(masterHostIpPort);
        String ip = ipOf(masterHostIpPort);
        String channelKey = ip + ":" + grpcPort;
        ManagedChannel channel = channels.computeIfAbsent(
                channelKey, ignored -> createChannel(ip, grpcPort));
        if (shutdown.get()) {
            channels.remove(channelKey, channel);
            channel.shutdownNow();
            throw Status.UNAVAILABLE
                    .withDescription("FlexLB gRPC forwarder is shutting down")
                    .asRuntimeException();
        }
        return channel;
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

        // SELF_TARGET is expected briefly while the cached master converges.
        // Its tagged metric is sufficient; retain a request log only for an
        // actual hop-limit violation.
        if (blockReason == ForwardBlockReason.HOP_LIMIT) {
            Logger.warn(
                    "event=flexlb_forward_blocked request_id={} operation={} reason={} "
                            + "forward_hop={} local_ip={} cached_master={} is_master={}",
                    requestId, operation.logValue(), blockReason.name(), incomingHop,
                    localIp, masterHostIpPort, lbStatusConsistencyService.isMaster());
        }
        reportForwardResult(ipOfOrLocal(masterHostIpPort), blockReason.name());
        return guard;
    }

    private enum ForwardOperation {
        SCHEDULE("schedule"),
        CANCEL("cancel"),
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
        if (!shutdown.compareAndSet(false, true)) {
            return;
        }
        for (ManagedChannel channel : channels.values()) {
            channel.shutdownNow();
        }
        channels.clear();
    }
}
