package org.flexlb.httpserver;

import io.grpc.Context;
import io.grpc.Status;
import io.grpc.stub.StreamObserver;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.schedule.grpc.FlexlbServiceGrpc;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol;
import org.flexlb.enums.ScheduleModeEnum;
import org.flexlb.interceptor.GrpcServerTimingInterceptor;
import org.flexlb.service.RouteService;
import org.flexlb.service.grace.ActiveRequestCounter;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;

@Component
public class FlexlbServiceImpl extends FlexlbServiceGrpc.FlexlbServiceImplBase {

    private final RouteService routeService;
    private final LBStatusConsistencyService lbStatusConsistencyService;
    private final EngineHealthReporter engineHealthReporter;
    private final ActiveRequestCounter activeRequestCounter;
    private final FlexlbGrpcForwarder grpcForwarder;
    private final ConfigService configService;
    private final BatchSchedulerReporter batchSchedulerReporter;
    private final ServerScheduleLatencyRecorder serverLatencyRecorder;
    public FlexlbServiceImpl(RouteService routeService,
                             LBStatusConsistencyService lbStatusConsistencyService,
                             EngineHealthReporter engineHealthReporter,
                             ActiveRequestCounter activeRequestCounter,
                             FlexlbGrpcForwarder grpcForwarder,
                             ConfigService configService,
                             BatchSchedulerReporter batchSchedulerReporter,
                             ServerScheduleLatencyRecorder serverLatencyRecorder) {
        this.routeService = routeService;
        this.lbStatusConsistencyService = lbStatusConsistencyService;
        this.engineHealthReporter = engineHealthReporter;
        this.activeRequestCounter = activeRequestCounter;
        this.grpcForwarder = grpcForwarder;
        this.configService = configService;
        this.batchSchedulerReporter = batchSchedulerReporter;
        this.serverLatencyRecorder = serverLatencyRecorder;
    }

    @Override
    public void schedule(FlexlbScheduleProtocol.FlexlbScheduleRequestPB request,
                         StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> responseObserver) {
        Long interceptedEntryNanos = GrpcServerTimingInterceptor.getNanos();
        serverLatencyRecorder.recordArrival(
                interceptedEntryNanos != null ? interceptedEntryNanos : System.nanoTime());
        ActiveRequestCounter.RequestToken token = activeRequestCounter.acquire();
        AtomicBoolean responded = new AtomicBoolean(false);
        BalanceContext ctx = null;

        try {
            ctx = buildContext(request);
            BalanceContext requestContext = ctx;
            boolean forwardToMaster = shouldForwardToMaster();
            engineHealthReporter.reportArriveDelayTime(requestContext);

            if (forwardToMaster) {
                FlexlbScheduleProtocol.FlexlbScheduleResponsePB forwardResponse =
                        grpcForwarder.forwardToMaster(request);
                if (forwardResponse != null) {
                    responded.set(true);
                    completeSchedule(requestContext, forwardResponse, responseObserver);
                    token.close();
                    return;
                }
            }

            CompletableFuture<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> routeFuture =
                    routeLocally(requestContext);

            Context grpcContext = Context.current();
            Context.CancellationListener cancellationListener = context -> {
                    requestContext.cancel();
                    routeFuture.completeExceptionally(
                            Status.CANCELLED.withDescription("gRPC context cancelled").asRuntimeException());
            };
            grpcContext.addListener(cancellationListener, Runnable::run);

            routeFuture.whenComplete((response, ex) -> {
                grpcContext.removeListener(cancellationListener);
                try {
                    if (responded.compareAndSet(false, true)) {
                        if (ex != null) {
                            Logger.error("FlexlbService.schedule async error, request_id={}", request.getRequestId(), ex);
                            completeSchedule(requestContext, buildErrorResponse(ex), responseObserver);
                        } else {
                            completeSchedule(requestContext, response, responseObserver);
                        }
                    }
                } catch (Exception e) {
                    Logger.error("FlexlbService.schedule callback error, request_id={}", request.getRequestId(), e);
                } finally {
                    token.close();
                }
            });

        } catch (Exception e) {
            Logger.error("FlexlbService.schedule error, request_id={}", request.getRequestId(), e);
            try {
                if (responded.compareAndSet(false, true)) {
                    completeSchedule(ctx, buildErrorResponse(e), responseObserver);
                }
            } catch (Exception inner) {
                Logger.error("FlexlbService.schedule error-response send failed, request_id={}",
                             request.getRequestId(), inner);
            } finally {
                token.close();
            }
        }
    }

    private CompletableFuture<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> routeLocally(BalanceContext ctx) {
        return routeService.route(ctx).thenApply(this::toProtoResponse);
    }

    private void completeSchedule(BalanceContext ctx,
                                  FlexlbScheduleProtocol.FlexlbScheduleResponsePB response,
                                  StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer) {
        // Report ACK-to-response time for BATCH path (only when engine ACK was received)
        if (ctx != null && ctx.getAckAtMs() > 0) {
            long ackToResponseMs = System.currentTimeMillis() - ctx.getAckAtMs();
            String prefillIp = "";
            String prefillIpPort = "";
            if (ctx.getResponse() != null && ctx.getResponse().getServerStatus() != null) {
                for (ServerStatus ss : ctx.getResponse().getServerStatus()) {
                    if (ss.getRole() == RoleType.PREFILL) {
                        prefillIp = ss.getServerIp() != null ? ss.getServerIp() : "";
                        prefillIpPort = ss.getServerIp() != null
                                ? ss.getServerIp() + ":" + ss.getHttpPort() : "";
                        break;
                    }
                }
            }
            batchSchedulerReporter.reportAckToResponseTimeMs(
                    RoleType.PREFILL.name(), prefillIp, prefillIpPort, ackToResponseMs);
        }
        try {
            observer.onNext(response);
            observer.onCompleted();
        } finally {
            serverLatencyRecorder.recordCompletion(ctx, System.nanoTime());
        }
        if (ctx != null) {
            ctx.setSuccess(response.getSuccess());
            if (!response.getSuccess()) {
                ctx.setErrorMessage(response.getErrorMessage());
            }
            engineHealthReporter.reportBalancingService(ctx);
        }
    }

    private FlexlbScheduleProtocol.FlexlbScheduleResponsePB buildErrorResponse(Throwable error) {
        return buildErrorResponse(500,
                error.getMessage() != null ? error.getMessage() : "internal error");
    }

    private FlexlbScheduleProtocol.FlexlbScheduleResponsePB buildErrorResponse(int code, String message) {
        return FlexlbScheduleProtocol.FlexlbScheduleResponsePB.newBuilder()
                .setSuccess(false)
                .setCode(code)
                .setErrorMessage(message)
                .build();
    }

    private BalanceContext buildContext(FlexlbScheduleProtocol.FlexlbScheduleRequestPB pb) {
        BalanceContext ctx = new BalanceContext();

        Request request = new Request();
        request.setRequestId(pb.getRequestId());
        request.setBlockCacheKeys(pb.getBlockCacheKeysList());
        request.setSeqLen(pb.getSeqLen());
        request.setGenerateTimeout(pb.getGenerateTimeout());
        request.setRequestTimeMs(pb.getRequestTimeMs());
        request.setMaxNewTokens(pb.getMaxNewTokens());
        request.setNumBeams(pb.getNumBeams());
        request.setForceDisableSpRun(pb.getForceDisableSpRun());
        request.setModel(pb.getModel());
        request.setApiKey(pb.getApiKey());
        request.setCacheKeyBlockSize(pb.getCacheKeyBlockSize());
        ctx.setRequest(request);

        if (!pb.getGenerateInput().isEmpty()) {
            ctx.setGenerateInputPbBytes(pb.getGenerateInput().toByteArray());
        }

        ctx.setScheduleMode(resolveScheduleMode(pb.getScheduleMode(), configService.loadBalanceConfig()));

        // Capture gRPC server entry time from interceptor context for delay metric splitting
        Long grpcEntryTime = GrpcServerTimingInterceptor.get();
        if (grpcEntryTime != null) {
            ctx.setGrpcEntryTime(grpcEntryTime);
        }
        Long grpcEntryNanos = GrpcServerTimingInterceptor.getNanos();
        if (grpcEntryNanos != null) {
            ctx.setGrpcEntryNanos(grpcEntryNanos);
        }

        return ctx;
    }

    private static ScheduleModeEnum resolveScheduleMode(FlexlbScheduleProtocol.FlexlbScheduleModePB mode,
                                                        FlexlbConfig config) {
        return switch (mode) {
            case FLEXLB_SCHEDULE_BATCH -> ScheduleModeEnum.BATCH;
            case FLEXLB_SCHEDULE_DIRECT -> ScheduleModeEnum.DIRECT;
            case FLEXLB_SCHEDULE_QUEUE -> ScheduleModeEnum.QUEUE;
            default -> config.getDefaultScheduleModeEnum();
        };
    }

    private FlexlbScheduleProtocol.FlexlbScheduleResponsePB toProtoResponse(Response response) {
        FlexlbScheduleProtocol.FlexlbScheduleResponsePB.Builder builder =
                FlexlbScheduleProtocol.FlexlbScheduleResponsePB.newBuilder();
        if (response == null) {
            return builder.setSuccess(false).setCode(500).setErrorMessage("null response").build();
        }
        builder.setSuccess(response.isSuccess());
        builder.setCode(response.getCode());
        if (response.getErrorMessage() != null) {
            builder.setErrorMessage(response.getErrorMessage());
        }
        if (response.getRealMasterHost() != null) {
            builder.setRealMasterHost(response.getRealMasterHost());
        }
        builder.setQueueLength(response.getQueueLength() != null ? response.getQueueLength() : 0);
        builder.setEnqueuedByMaster(response.isEnqueuedByMaster());

        if (response.getServerStatus() != null) {
            for (ServerStatus ss : response.getServerStatus()) {
                builder.addServerStatus(FlexlbScheduleProtocol.FlexlbServerStatusPB.newBuilder()
                        .setRole(ss.getRole().getCode())
                        .setServerIp(ss.getServerIp() != null ? ss.getServerIp() : "")
                        .setHttpPort(ss.getHttpPort())
                        .setGrpcPort(ss.getGrpcPort())
                        .build());
            }
        }
        return builder.build();
    }

    private boolean shouldForwardToMaster() {
        return lbStatusConsistencyService.isNeedConsistency()
                && !lbStatusConsistencyService.isMaster();
    }
}
