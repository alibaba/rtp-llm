package org.flexlb.mock.engine.rtp;

import io.grpc.*;
import io.grpc.stub.StreamObserver;
import java.util.*;
import java.util.concurrent.*;
import org.flexlb.engine.grpc.EngineRpcService.*;

/** Single-DP native envelope; admission is acknowledged independently of Fetch. */
final class PrefillBatch {
    static void enqueue(PrefillGeneration generation, EnqueueBatchRequestPB request, StreamObserver<EnqueueBatchResponsePB> observer) {
        var seen = new HashSet<Long>();
        boolean duplicate = false;
        int size = 0;
        for (var slot : request.getDpSlotsList()) for (var item : slot.getRequestsList()) {
            size++;
            if (item.hasInput() && !seen.add(item.getInput().getRequestId())) duplicate = true;
        }
        var pending = new ArrayList<CompletableFuture<EnqueueBatchResponsePB>>();
        var inputs = new ArrayList<GenerateInputPB>();
        var admissions = new ArrayList<CompletableFuture<Void>>();
        for (var slot : request.getDpSlotsList()) for (var item : slot.getRequestsList()) {
            long id = item.hasInput() ? item.getInput().getRequestId() : 0;
            Throwable invalid = !item.hasInput() ? Status.INVALID_ARGUMENT.withDescription("Missing batch input").asRuntimeException()
                : duplicate ? Status.ALREADY_EXISTS.withDescription("Duplicate request_id in batch").asRuntimeException()
                : slot.getDpRank() != 0 ? Status.INVALID_ARGUMENT.withDescription("Mock supports only dp_rank 0").asRuntimeException() : null;
            var admission = new CompletableFuture<Void>();
            if (invalid != null) admission.completeExceptionally(invalid);
            else {
                inputs.add(item.getInput().toBuilder().setGroupSize(size)
                    .setGroupId(com.google.protobuf.Int64Value.of(request.getBatchId())).build());
                admissions.add(admission);
            }
            pending.add(admission.handle((unused, error) -> {
                var result = EnqueueBatchResponsePB.newBuilder();
                if (error == null) result.addSuccesses(EnqueueBatchSuccessPB.newBuilder().setRequestId(id));
                else result.addErrors(EnqueueBatchErrorPB.newBuilder().setRequestId(id).setErrorInfo(details(error)));
                return result.build();
            }));
        }
        var prepared = generation.enqueue(inputs, request.getFetchAttachTimeoutMs());
        for (int i = 0; i < admissions.size(); i++) {
            var admission = admissions.get(i);
            prepared.get(i).whenComplete((unused, error) -> {
                if (error == null) admission.complete(null); else admission.completeExceptionally(error);
            });
        }
        CompletableFuture.allOf(pending.toArray(CompletableFuture[]::new)).thenRun(() -> {
            var result = EnqueueBatchResponsePB.newBuilder().setBatchId(request.getBatchId());
            pending.forEach(p -> { var value = p.join(); result.addAllSuccesses(value.getSuccessesList()).addAllErrors(value.getErrorsList()); });
            observer.onNext(result.build()); observer.onCompleted();
        });
    }

    private static ErrorDetailsPB details(Throwable error) {
        while (error instanceof CompletionException && error.getCause() != null) error = error.getCause();
        var trailers = Status.trailersFromThrowable(error);
        if (trailers != null) {
            var raw = trailers.get(Metadata.Key.of("grpc-status-details-bin", Metadata.BINARY_BYTE_MARSHALLER));
            if (raw != null) try { return ErrorDetailsPB.parseFrom(raw); }
            catch (com.google.protobuf.InvalidProtocolBufferException malformed) { /* Fall back to gRPC status. */ }
        }
        var status = Status.fromThrowable(error);
        return ErrorDetailsPB.newBuilder().setErrorCode(status.getCode().value())
            .setErrorMessage(status.getDescription() == null ? status.getCode().name() : status.getDescription()).build();
    }
}
