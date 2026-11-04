package org.flexlb.balance.scheduler;

import com.google.protobuf.ByteString;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import com.google.protobuf.InvalidProtocolBufferException;
import com.google.protobuf.UnknownFieldSet;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.DecodeResources.AdmissionCapacity;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.RequestTimeEstimation.WorkSnapshot;
import org.flexlb.engine.grpc.EngineRpcService.GenerateInputPB;
import org.flexlb.util.Failures;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;

/** One delivery owns temporary resources from preparation through the dispatch handoff. */
@lombok.Getter
@lombok.experimental.Accessors(fluent = true)
public final class Delivery implements AutoCloseable {
    long batchId;
    List<Member> members;
    WorkSnapshot precedingWork;
    long predictedMs;
    PrefillTimePredictor.PrefillBatchFeatures features;
    PrefillTimePredictor.Evaluator evaluator;
    String reason;
    int remainingQueueDepth;
    private boolean committed;
    private org.flexlb.balance.endpoint.WorkerEndpoint.GenerationPin generationPin;
    private boolean removedFailedMember;
    long selectionTimeMs;
    RequestRoute bypassedHead;
    DeliveryExecution.DeliveryPermit executionPermit;
    final PrefillEndpoint endpoint;
    final boolean batch;
    final long[] predictions;
    PrefillEndpoint.BatchReservation reservation;
    CapacityBoundary boundary;
    RequestRoute blocked;

    public Delivery(
            PrefillEndpoint endpoint,
            boolean batch,
            int size,
            PrefillTimePredictor.Evaluator evaluator,
            String reason) {
        this.endpoint = endpoint;
        this.batch = batch;
        this.evaluator = evaluator;
        this.reason = reason;
        members = new ArrayList<>(size);
        predictions = batch ? null : new long[size];
    }

    /** Endpoint calls this only after validating every member under its resource lock. */
    public void committed(
            org.flexlb.balance.endpoint.WorkerEndpoint.GenerationPin pin,
            WorkSnapshot precedingWork,
            int remainingQueueDepth,
            boolean removedFailedMember) {
        checkState(!committed, "delivery already committed");
        this.generationPin = Objects.requireNonNull(pin);
        this.precedingWork = precedingWork;
        this.remainingQueueDepth = remainingQueueDepth;
        this.removedFailedMember = removedFailedMember;
        committed = true;
    }

    public List<RequestRoute> items() {
        return members.stream().map(Member::route).toList();
    }

    public RequestRoute failedMember() {
        return boundary != null && boundary.status() == CapacityBoundary.Status.FAILED
                ? blocked
                : null;
    }

    /** End the sender's temporary ownership; committed ledger rows settle through Lifecycle. */
    Throwable releasePreparation() {
        Throwable failure = null;
        for (Member member : members) failure = Failures.run(failure, member::close);
        if (!committed && reservation != null)
            failure = Failures.run(failure, () -> endpoint.rollback(reservation));
        var pin = generationPin;
        if (pin != null) {
            failure = Failures.append(failure, Failures.close(pin));
            generationPin = null;
        }
        return failure;
    }

    @Override
    public void close() {
        Throwable failure = releasePreparation();
        if (executionPermit != null)
            failure = Failures.append(failure, Failures.close(executionPermit));
        Failures.rethrow(failure, "temporary delivery resource cleanup failed");
    }

    public static final class Member implements AutoCloseable {
        @lombok.Getter
        @lombok.experimental.Accessors(fluent = true)
        private final RequestRoute route;

        @lombok.Getter
        @lombok.experimental.Accessors(fluent = true)
        private final DecodeEndpoint.EngineDispatchPermit decode;

        @lombok.Getter
        @lombok.experimental.Accessors(fluent = true)
        private Delivery.SendMode kind = Delivery.SendMode.NONE;

        @lombok.Getter
        @lombok.experimental.Accessors(fluent = true)
        private long batchId;

        public Member(RequestRoute route, DecodeEndpoint.EngineDispatchPermit decode) {
            this.route = Objects.requireNonNull(route);
            this.decode = decode;
        }

        void initialize(Delivery.SendMode kind, long batchId) {
            checkState(this.kind == Delivery.SendMode.NONE, "delivery member already claimed");
            checkArgument(
                    kind != Delivery.SendMode.NONE
                            && (kind != Delivery.SendMode.BATCH_ENQUEUE || batchId > 0L),
                    "invalid delivery identity");
            this.kind = Objects.requireNonNull(kind);
            this.batchId = batchId;
        }

        public RequestContext owner() {
            return route.ctx();
        }

        Delivery.Member claimDelivery(Delivery.SendMode kind, long batchId) {
            return DeliveryDispatcher.commitDelivery(
                    owner().data().getLifecycle(), this, kind, batchId);
        }

        void recordFailure(Throwable failure) {
            route.ctx().data().getLifecycle().tasks.recordFailure(failure);
        }

        static CapacityBoundary.Attempt<Member> prepare(
                RequestRoute route, AdmissionCapacity capacity) {
            var endpoint = route.decodeEp();
            var reservation = route.decodeReservation();
            if (endpoint == null && reservation == null) {
                return CapacityBoundary.Attempt.accepted(new Member(route, null));
            }
            if (endpoint == null || reservation == null) {
                return CapacityBoundary.Attempt.rejected(
                        CapacityBoundary.failed(
                                new IllegalStateException(
                                        "Decode reservation is unavailable: request_id="
                                                + route.requestId())));
            }
            DecodeEndpoint.EngineDispatchPermit permit = null;
            try {
                var acquisition = endpoint.acquireDispatchPermit(reservation, capacity);
                switch (acquisition.status()) {
                    case ACQUIRED, ALREADY_ACCEPTED -> permit = acquisition.handle();
                    case CAPACITY_FULL -> {
                        return CapacityBoundary.Attempt.rejected(
                                CapacityBoundary.unavailable(
                                        CapacityBoundary.Resource.DECODE_ENGINE));
                    }
                    case NOT_OWNED, NOT_QUEUED -> {
                        return CapacityBoundary.Attempt.rejected(CapacityBoundary.OWNERSHIP_LOST);
                    }
                    case ENDPOINT_RETIRED, ALREADY_ACQUIRED -> {
                        String reason =
                                acquisition.status()
                                                == DecodeResources.EngineDispatchPermitAcquireStatus
                                                        .ENDPOINT_RETIRED
                                        ? "Decode endpoint generation retired: request_id="
                                        : "Decode dispatch permit already acquired: request_id=";
                        return CapacityBoundary.Attempt.rejected(
                                CapacityBoundary.failed(
                                        new IllegalStateException(reason + route.requestId())));
                    }
                }
                return CapacityBoundary.Attempt.accepted(new Member(route, permit));
            } catch (Throwable failure) {
                if (permit != null) {
                    failure = Failures.run(failure, permit::release);
                }
                return CapacityBoundary.Attempt.rejected(CapacityBoundary.failed(failure));
            }
        }

        @Override
        public void close() {
            if (decode != null) {
                decode.release();
            }
        }

        private volatile java.util.concurrent.CompletableFuture<Void> completion;

        @SuppressWarnings("rawtypes")
        private static final java.util.concurrent.atomic.AtomicReferenceFieldUpdater<
                        Delivery.Member, java.util.concurrent.CompletableFuture>
                COMPLETION =
                        java.util.concurrent.atomic.AtomicReferenceFieldUpdater.newUpdater(
                                Delivery.Member.class,
                                java.util.concurrent.CompletableFuture.class,
                                "completion");

        private java.util.concurrent.CompletableFuture<Void> completionFuture() {
            var current = completion;
            if (current != null) return current;
            var created = new java.util.concurrent.CompletableFuture<Void>();
            return COMPLETION.compareAndSet(this, null, created) ? created : completion;
        }

        public java.util.concurrent.CompletionStage<Void> completion() {
            return completionFuture().minimalCompletionStage();
        }

        void complete() {
            completionFuture().complete(null);
        }
    }

    /** Validate input without allocating a decoded token array; Engine sees the same protobuf wire fields. */
    static GenerateInputPB parseInput(ByteString bytes) throws InvalidProtocolBufferException {
        try {
            CodedInputStream input = bytes.newCodedInput();
            var metadata = GenerateInputPB.newBuilder();
            var tokens = new Tokens();
            int retainedStart = 0;
            int tagStart = 0;
            for (int tag; (tag = input.readTag()) != 0; tagStart = input.getTotalBytesRead()) {
                if (tag == 16 || tag == 18) {
                    mergeMetadata(metadata, bytes, retainedStart, tagStart);
                    tokens.readField(tag, input, bytes);
                    retainedStart = input.getTotalBytesRead();
                    continue;
                }
                if (!input.skipField(tag)) {
                    input.checkLastTagWas(0);
                }
            }
            input.checkLastTagWas(0);
            mergeMetadata(metadata, bytes, retainedStart, bytes.size());
            var parsed = metadata.build();
            ByteString encoded = tokens.finish();
            if (!encoded.isEmpty() && bytes.asReadOnlyByteBufferList().size() > 1) {
                // A segmented input may use a copying stream decoder for metadata. Give
                // tokens their own compact storage so they cannot pin copied large fields.
                var compact = ByteString.newOutput(encoded.size());
                encoded.writeTo(compact);
                encoded = compact.toByteString();
            }
            if (!encoded.isEmpty()) {
                // Unknown fields use the standard protobuf writer. On the Engine this is the
                // known packed token_ids field again; no schema or RPC method changes.
                var field = UnknownFieldSet.Field.newBuilder().addLengthDelimited(encoded).build();
                parsed =
                        parsed.toBuilder()
                                .mergeUnknownFields(
                                        UnknownFieldSet.newBuilder().mergeField(2, field).build())
                                .build();
            }
            return parsed;
        } catch (InvalidProtocolBufferException malformed) {
            // Diagnostics can outlive dispatch. Never let a parse failure pin a partial
            // tensor/message.
            throw malformed.setUnfinishedMessage(null);
        } catch (IOException failure) {
            throw new InvalidProtocolBufferException(failure);
        }
    }

    private static void mergeMetadata(
            GenerateInputPB.Builder metadata, ByteString bytes, int start, int end)
            throws IOException {
        if (start == end) {
            return;
        }
        var input = bytes.substring(start, end).newCodedInput();
        // ByteString is immutable: tensors and unknown byte fields can share its storage.
        input.enableAliasing(true);
        metadata.mergeFrom(input);
        input.checkLastTagWas(0);
    }

    private static final class Tokens {
        private ByteString packed;
        private ByteString.Output combined;
        private CodedOutputStream writer;

        private void readField(int tag, CodedInputStream input, ByteString bytes)
                throws IOException {
            if (tag == 16) {
                writer().writeInt32NoTag(input.readInt32());
                return;
            }
            int length = input.readRawVarint32();
            int previousLimit = input.pushLimit(length);
            int start = input.getTotalBytesRead();
            while (input.getBytesUntilLimit() > 0) {
                input.readInt32();
            }
            input.popLimit(previousLimit);
            ByteString chunk = bytes.substring(start, input.getTotalBytesRead());
            if (packed == null && writer == null) {
                packed = chunk;
            } else {
                writer().writeRawBytes(chunk);
            }
        }

        private CodedOutputStream writer() throws IOException {
            if (writer == null) {
                combined = ByteString.newOutput(packed == null ? 32 : Math.max(32, packed.size()));
                writer = CodedOutputStream.newInstance(combined, 64);
                if (packed != null) {
                    writer.writeRawBytes(packed);
                    packed = null;
                }
            }
            return writer;
        }

        private ByteString finish() throws IOException {
            if (writer == null) {
                return packed == null ? ByteString.EMPTY : packed;
            }
            writer.flush();
            return combined.toByteString();
        }
    }

    /** Transport result for one exact delivery claim. */
    public record Result(Status status, Throwable cause) {

        public Result {
            checkArgument(status != null, "delivery status is required");
            checkArgument(
                    (status == Status.DELIVERED) != (cause != null),
                    "only unsuccessful delivery requires a cause");
        }

        public static Result delivered() {
            return new Result(Status.DELIVERED, null);
        }

        public static Result notSent(Throwable cause) {
            return new Result(Status.NOT_SENT, cause);
        }

        public static Result prefillRejected(Throwable cause) {
            return new Result(Status.PREFILL_REJECTED, cause);
        }

        public static Result uncertain(Throwable cause) {
            return new Result(Status.UNCERTAIN, cause);
        }

        public boolean failed() {
            return status() == Status.PREFILL_REJECTED || status() == Status.NOT_SENT;
        }

        public enum Status {
            DELIVERED,
            /** Local failure before RPC invocation; no remote work was started by this claim. */
            NOT_SENT,
            /** Final per-member EnqueueBatch error; Decode may already own resources. */
            PREFILL_REJECTED,
            UNCERTAIN
        }
    }

    /** Whether unavailable capacity may shorten an already ordered selection. */
    public enum PrefixPolicy {
        ALL_SELECTED,
        ORDERED_PREFIX
    }

    /** Delivery mode, also used by request diagnostics before a delivery is claimed. */
    public enum SendMode {
        /** The request is still wholly owned by the scheduler. */
        NONE,

        /** An EnqueueBatch operation may have made the request visible to the engine. */
        BATCH_ENQUEUE,

        /** The route decision may have been returned to the caller. */
        ROUTE_DECISION
    }
}
