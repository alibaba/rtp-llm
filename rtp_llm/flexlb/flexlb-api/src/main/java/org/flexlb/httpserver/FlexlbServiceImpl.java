package org.flexlb.httpserver;

import io.grpc.stub.StreamObserver;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.balance.scheduler.RequestLifecycleSnapshot;
import org.flexlb.config.PrioritySloPolicy;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.schedule.grpc.FlexlbServiceGrpc;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol;
import org.flexlb.interceptor.GrpcQosHeaderInterceptor;
import org.flexlb.interceptor.GrpcServerTimingInterceptor;
import org.flexlb.service.RouteService;
import org.flexlb.service.grace.ActiveRequestCounter;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.service.monitor.PrioritySchedulerReporter;
import org.flexlb.config.ConfigService;
import org.flexlb.util.Logger;
import org.flexlb.util.PriorityNormalizer;
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
    private final PrioritySloPolicy prioritySloPolicy;
    private final PrioritySchedulerReporter prioritySchedulerReporter;
    public FlexlbServiceImpl(RouteService routeService,
                             LBStatusConsistencyService lbStatusConsistencyService,
                             EngineHealthReporter engineHealthReporter,
                             ActiveRequestCounter activeRequestCounter,
                             FlexlbGrpcForwarder grpcForwarder,
                             ConfigService configService,
                             BatchSchedulerReporter batchSchedulerReporter,
                             ServerScheduleLatencyRecorder serverLatencyRecorder,
                             PrioritySloPolicy prioritySloPolicy,
                             PrioritySchedulerReporter prioritySchedulerReporter) {
        this.routeService = routeService;
        this.lbStatusConsistencyService = lbStatusConsistencyService;
        this.engineHealthReporter = engineHealthReporter;
        this.activeRequestCounter = activeRequestCounter;
        this.grpcForwarder = grpcForwarder;
        this.configService = configService;
        this.batchSchedulerReporter = batchSchedulerReporter;
        this.serverLatencyRecorder = serverLatencyRecorder;
        this.prioritySloPolicy = prioritySloPolicy;
        this.prioritySchedulerReporter = prioritySchedulerReporter;
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

            routeFuture.whenComplete((response, ex) -> {
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

    @Override
    public void getRequestState(FlexlbScheduleProtocol.GetRequestStateRequestPB request,
                                StreamObserver<FlexlbScheduleProtocol.GetRequestStateResponsePB> responseObserver) {
        if (shouldForwardToMaster()) {
            FlexlbScheduleProtocol.GetRequestStateResponsePB forwarded =
                    grpcForwarder.forwardGetRequestStateToMaster(request);
            if (forwarded != null && forwarded.getFound()) {
                responseObserver.onNext(forwarded);
                responseObserver.onCompleted();
                return;
            }
        }
        RequestLifecycleSnapshot snapshot = routeService.getRequestState(
                request.getRequestId(), request.getBatchId());
        FlexlbScheduleProtocol.GetRequestStateResponsePB.Builder response =
                FlexlbScheduleProtocol.GetRequestStateResponsePB.newBuilder().setFound(snapshot != null);
        if (snapshot != null) {
            response.setLifecycle(toLifecycleProto(snapshot));
        }
        responseObserver.onNext(response.build());
        responseObserver.onCompleted();
    }

    private CompletableFuture<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> routeLocally(BalanceContext ctx) {
        return routeService.route(ctx).thenApply(response -> {
            FlexlbScheduleProtocol.FlexlbScheduleResponsePB.Builder builder =
                    toProtoResponse(response).toBuilder();
            RequestLifecycleSnapshot lifecycle = routeService.getRequestState(ctx.getRequestId(), 0);
            if (lifecycle != null) {
                builder.setLifecycle(toLifecycleProto(lifecycle));
            }
            return builder.build();
        });
    }

    private void completeSchedule(BalanceContext ctx,
                                  FlexlbScheduleProtocol.FlexlbScheduleResponsePB response,
                                  StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer) {
        // Report ACK-to-response time for BATCH path (only when engine ACK was received)
        if (ctx != null && ctx.getAckAtMs() > 0) {
            long ackToResponseMs = System.currentTimeMillis() - ctx.getAckAtMs();
            String prefillIp = "";
            if (ctx.getResponse() != null && ctx.getResponse().getServerStatus() != null) {
                for (ServerStatus ss : ctx.getResponse().getServerStatus()) {
                    if (ss.getRole() == RoleType.PREFILL) {
                        prefillIp = ss.getServerIp() != null ? ss.getServerIp() : "";
                        break;
                    }
                }
            }
            batchSchedulerReporter.reportAckToResponseTimeMs(
                    RoleType.PREFILL.name(), prefillIp, ackToResponseMs);
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
            reportAutoTpmSchedule(ctx, response);
        }
    }

    /**
     * Auto-TPM Phase 0 observability: per-request one-line schedule log plus
     * {@code auto_tpm.schedule.latency_ms}. Shared by the legacy path and the
     * priority scheduler path (both funnel through completeSchedule).
     */
    private void reportAutoTpmSchedule(BalanceContext ctx,
                                       FlexlbScheduleProtocol.FlexlbScheduleResponsePB response) {
        try {
            long now = System.currentTimeMillis();
            long latencyMs = now - ctx.getStartTime();
            boolean success = response.getSuccess();
            String result = success ? "success" : "error_" + response.getCode();
            prioritySchedulerReporter.reportScheduleLatency(ctx.getPriority(), result, latencyMs);
            // Approximate TTFT: schedule-complete minus arrival, as seen by
            // FlexLB. The true TTFT (first token emitted by the engine) is not
            // observable here, so this proxy omits engine-side prefill
            // execution time (task10 P2-8).
            prioritySchedulerReporter.reportTtft(ctx.getPriority(), latencyMs);
            if (ctx.getDeadlineMs() > 0 && now > ctx.getDeadlineMs()) {
                prioritySchedulerReporter.reportDeadlineMiss(ctx.getPriority());
            }

            String selectedPrefill = "";
            String selectedDecode = "";
            if (ctx.getResponse() != null && ctx.getResponse().getServerStatus() != null) {
                for (ServerStatus ss : ctx.getResponse().getServerStatus()) {
                    if (ss.getRole() == RoleType.PREFILL || ss.getRole() == RoleType.PDFUSION) {
                        selectedPrefill = ss.getServerIp() != null ? ss.getServerIp() : "";
                    } else if (ss.getRole() == RoleType.DECODE) {
                        selectedDecode = ss.getServerIp() != null ? ss.getServerIp() : "";
                    }
                }
            }
            // Metrics above stay always-on; the per-request log line drops to
            // DEBUG when Auto-TPM is disabled to avoid INFO noise on the
            // legacy path (task10 P2-7).
            String logFormat = "[auto-tpm] request_id={} priority={} seq_len={} max_new_tokens={} "
                    + "request_slo_ms={} deadline_ms={} schedule_attempt={} plan_type={} plan_cost={} "
                    + "victim_count={} selected_prefill={} selected_decode={} failure_reason={} commit_result={}";
            Object[] logArgs = {
                    ctx.getRequestId(), ctx.getPriority(), ctx.getRequest().getSeqLen(),
                    ctx.getRequest().getMaxNewTokens(),
                    ctx.getRequestSloMs(), ctx.getDeadlineMs(),
                    ctx.getScheduleAttempt(), ctx.getPlanType(), ctx.getPlanCost(), ctx.getVictimCount(),
                    selectedPrefill, selectedDecode,
                    success ? "" : response.getErrorMessage(),
                    result};
            if (configService.loadBalanceConfig().isAutoTpmEnabled()) {
                Logger.info(logFormat, logArgs);
            } else {
                Logger.debug(logFormat, logArgs);
            }
        } catch (Exception e) {
            Logger.warn("[auto-tpm] schedule observability report failed, request_id={}",
                    ctx.getRequestId(), e);
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

        // Auto-TPM: normalize priority (proto valid value > metadata header > default)
        // and derive the per-request SLO / coarse deadline for observability.
        // normalize() always returns 1-100 (never NO_PRIORITY/0): a request
        // carrying no priority gets the default (50) and participates in
        // Auto-TPM at the normal level, so the SLO block runs unconditionally.
        int priority = PriorityNormalizer.normalize(pb.getPriority(),
                GrpcQosHeaderInterceptor.get(),
                configService.loadBalanceConfig().getAutoTpmDefaultPriority());
        request.setPriority(priority);
        ctx.setRequest(request);

        long requestSloMs = prioritySloPolicy.requestSloMs(pb.getSeqLen(), priority);
        ctx.setRequestSloMs(requestSloMs);
        // Coarse deadline (no predicted prefill time yet); the priority
        // scheduler overwrites it once the target prefill endpoint is known.
        ctx.setDeadlineMs(ctx.getStartTime() + requestSloMs);
        prioritySchedulerReporter.reportRequest(priority,
                prioritySloPolicy.bucketLabel(pb.getSeqLen()), requestSloMs);

        if (!pb.getGenerateInput().isEmpty()) {
            ctx.setGenerateInputPbBytes(pb.getGenerateInput().toByteArray());
        }

        ctx.setScheduleMode(configService.loadBalanceConfig().getDefaultScheduleModeEnum());

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

    private static FlexlbScheduleProtocol.RequestLifecyclePB toLifecycleProto(
            RequestLifecycleSnapshot snapshot) {
        FlexlbScheduleProtocol.RequestLifecyclePB.Builder lifecycle =
                FlexlbScheduleProtocol.RequestLifecyclePB.newBuilder()
                        .setRequestId(snapshot.requestId())
                        .setState(switch (snapshot.state()) {
                            case QUEUED -> FlexlbScheduleProtocol.RequestStatePB.REQUEST_STATE_QUEUED;
                            case DISPATCHING -> FlexlbScheduleProtocol.RequestStatePB.REQUEST_STATE_DISPATCHING;
                            case ACKNOWLEDGED -> FlexlbScheduleProtocol.RequestStatePB.REQUEST_STATE_ACKNOWLEDGED;
                            case TIMED_OUT -> FlexlbScheduleProtocol.RequestStatePB.REQUEST_STATE_TIMED_OUT;
                            case FAILED -> FlexlbScheduleProtocol.RequestStatePB.REQUEST_STATE_FAILED;
                            case COMPLETED -> FlexlbScheduleProtocol.RequestStatePB.REQUEST_STATE_COMPLETED;
                        });
        if (snapshot.batchId() > 0) {
            lifecycle.setBatchId(snapshot.batchId());
        }
        return lifecycle.build();
    }
}
