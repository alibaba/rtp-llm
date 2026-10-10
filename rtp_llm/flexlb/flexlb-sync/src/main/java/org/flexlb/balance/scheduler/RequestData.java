package org.flexlb.balance.scheduler;

import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;
import io.opentelemetry.context.Context;
import lombok.Getter;
import lombok.Setter;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.engine.grpc.EngineRpcService.GenerateInputPB;
import org.flexlb.util.Failures;

import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.FutureTask;

import static com.google.common.base.Preconditions.checkState;

/** Request input and telemetry. Lifecycle decisions belong to the exact RequestContext. */
@Getter
public final class RequestData {
    private final FlexlbConfig config;
    private final long startTime = System.currentTimeMillis();
    private final long serviceStartNanos = System.nanoTime();
    private volatile Request request;
    private volatile SchedulingMetadata schedulingMetadata;
    private volatile CompletableFuture<Response> future;
    private volatile AbstractRequestScheduler scheduler;
    @Getter(lombok.AccessLevel.NONE) private volatile FutureTask<GenerateInputPB> generateInput;
    private volatile int generateInputSize;
    @Getter(lombok.AccessLevel.NONE) private volatile boolean registered;
    @Setter private Context traceContext;
    @Setter private long grpcEntryTime;
    @Setter private long grpcEntryNanos;
    @Setter private long routeSubmittedNanos;
    @Setter private long batchDispatchedNanos;
    @Setter private long enqueueTime;
    private long firstWorkerEnqueueTime;
    private long workerEnqueueSequence;
    @Setter private volatile long placementSequence;
    private long ackAtMs;
    private long ackAtNanos;
    @Setter private volatile Map<String, Object> schedulingDiagnostics;
    @Setter private String planType = "";
    @Setter private long planCost;
    @Setter private int victimCount;

    public RequestData(FlexlbConfig config) { this.config = Objects.requireNonNull(config, "config"); }

    public synchronized void setRequest(Request value) {
        checkState(!registered || request == value, "registered request inputs are immutable");
        request = Objects.requireNonNull(value, "request");
    }

    public synchronized void setSchedulingMetadata(SchedulingMetadata value) {
        checkState(!registered || schedulingMetadata == value, "registered scheduling metadata is immutable");
        schedulingMetadata = value;
    }

    public synchronized void setFuture(CompletableFuture<Response> value) {
        checkState(!registered && !(value instanceof RequestFuture) || future == value,
                "registered request future cannot be replaced or rebound");
        future = value;
    }

    /** Binding happens under the Repository -> Context lock; no callbacks execute here. */
    synchronized void register(AbstractRequestScheduler owner, RequestFuture response) {
        checkState(!registered && scheduler == null, "request data already registered");
        Objects.requireNonNull(request, "request");
        Objects.requireNonNull(owner, "scheduler");
        Objects.requireNonNull(response, "response");
        if (schedulingMetadata == null) {
            schedulingMetadata = SchedulingMetadata.explicit(request.getRawPriority(), config.resolveExpiresAtMs(startTime));
        }
        scheduler = owner;
        future = response;
        registered = true;
    }

    public long getRequestId() { return request.getRequestId(); }
    public int getPriority() { return schedulingMetadata == null ? request.getRawPriority() : schedulingMetadata.priority(); }
    public long getRequestExpiresAtMs() {
        return schedulingMetadata == null ? config.resolveExpiresAtMs(startTime) : schedulingMetadata.expiresAtMs();
    }

    public Response getResponse() {
        var response = future;
        if (response == null) { return null; }
        try { return response.getNow(null); }
        catch (CompletionException | CancellationException failed) { return null; }
    }

    /** Parsing remains deferred until delivery, and FutureTask memoizes both value and failure. */
    public synchronized void setGenerateInputPb(ByteString bytes) {
        checkState(!registered, "registered delivery input is immutable");
        generateInputSize = bytes == null ? 0 : bytes.size();
        generateInput = bytes == null || bytes.isEmpty() ? null : new FutureTask<>(() -> GenerateInputPB.parseFrom(bytes));
    }

    boolean hasGenerateInput() { return generateInput != null; }
    void prepareGenerateInput() { var input = generateInput; if (input != null) { input.run(); } }

    GenerateInputPB getGenerateInput() throws InvalidProtocolBufferException, InterruptedException {
        var input = generateInput;
        if (input == null) { return null; }
        input.run();
        try { return input.get(); }
        catch (ExecutionException failure) {
            if (failure.getCause() instanceof InvalidProtocolBufferException malformed) { throw malformed; }
            throw Failures.propagate(failure.getCause(), "Generate input parsing failed");
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            throw interrupted;
        }
    }

    synchronized void initializeWorkerQueue(long nowMs, long sequence) {
        if (workerEnqueueSequence == 0L) {
            firstWorkerEnqueueTime = nowMs;
            workerEnqueueSequence = sequence;
        }
    }

    synchronized void acknowledge(long nowMs, long nowNanos) {
        if (ackAtMs == 0L) { ackAtMs = nowMs; ackAtNanos = nowNanos; }
    }
}
