package org.flexlb.mockengine;

import io.grpc.Context;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.Status;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.MultimodalRpcServiceGrpc;

import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

/** Prefill consumes the same image hashes used by frontend routing. Never called on D. */
final class MockVitClient implements AutoCloseable {
    private final Map<String, ManagedChannel> channels = new ConcurrentHashMap<>();

    EngineRpcService.GenerateInputPB prepare(EngineRpcService.GenerateInputPB input, long budgetMs) {
        if (input.hasV41Inputs()) throw Status.UNIMPLEMENTED.withDescription("packed V4.1 Mock ViT backend is not configured").asRuntimeException();
        if (input.getMultimodalInputsCount() == 0) return input;
        var address = input.getGenerateConfig().getRoleAddrsList().stream()
                .filter(a -> a.getRoleStr().equals("VIT") || a.getRole() == EngineRpcService.RoleAddrPB.RoleType.VIT)
                .findFirst().orElseThrow(() -> Status.FAILED_PRECONDITION.withDescription("image request has no VIT role address").asRuntimeException());
        if (budgetMs <= 0) throw Status.DEADLINE_EXCEEDED.asRuntimeException();
        String target = address.getIp() + ":" + address.getGrpcPort();
        var channel = channels.computeIfAbsent(target, ignored -> ManagedChannelBuilder
                .forAddress(address.getIp(), address.getGrpcPort()).usePlaintext()
                .maxInboundMessageSize(64 << 20).disableRetry().build());
        var request = EngineRpcService.MultimodalInputsPB.newBuilder().setRequestId(input.getRequestId())
                .addAllMultimodalInputs(input.getMultimodalInputsList()).setTimeoutMs(Math.min(budgetMs, 120000)).build();
        var response = MultimodalRpcServiceGrpc.newBlockingStub(channel)
                .withDeadlineAfter(Math.min(budgetMs, 120000), TimeUnit.MILLISECONDS).remoteMultimodalEmbedding(request);
        if (Context.current().isCancelled()) throw Status.CANCELLED.asRuntimeException();
        var features = response.getMultimodalEmbedding();
        long rows = response.getSplitSizeList().stream().mapToLong(Integer::longValue).sum();
        if (response.getSplitSizeCount() != input.getMultimodalInputsCount()
                || response.getSplitSizeList().stream().anyMatch(size -> size <= 0)
                || features.getShapeCount() != 2 || features.getShape(0) != rows
                || features.getDataType() != EngineRpcService.TensorPB.DataType.FP32
                || rows * features.getShape(1) * 4 != features.getFp32Data().size())
            throw Status.DATA_LOSS.withDescription("invalid Mock ViT embedding shape or split sizes").asRuntimeException();
        byte[] tensor = features.getFp32Data().toByteArray();
        int stride = Math.toIntExact(features.getShape(1) * 4);
        var hashes = new ArrayList<Integer>();
        for (int i = 0; i < rows; i++) hashes.add(MockFeatureHash.row(tensor, i * stride, stride));
        if (response.hasMultimodalFeatureHash()) {
            var wire = response.getMultimodalFeatureHash().getInt32Data().asReadOnlyByteBuffer().order(ByteOrder.LITTLE_ENDIAN);
            if (wire.remaining() != hashes.size() * 4) throw Status.DATA_LOSS.withDescription("invalid ViT hash size").asRuntimeException();
            for (int hash : hashes) if (wire.getInt() != hash)
                throw Status.DATA_LOSS.withDescription("ViT hash does not match embedding").asRuntimeException();
        }
        if (input.getMultimodalTokenLayout().getSpansCount() != response.getSplitSizeCount())
            throw Status.FAILED_PRECONDITION.withDescription("frontend must provide expanded multimodal token layout").asRuntimeException();
        int cursor = 0, previousEnd = 0;
        for (int i = 0; i < response.getSplitSizeCount(); i++) {
            var span = input.getMultimodalTokenLayout().getSpans(i);
            if (span.getLength() != response.getSplitSize(i) || span.getOffset() < previousEnd
                    || (long)span.getOffset() + span.getLength() > input.getTokenIdsCount())
                throw Status.DATA_LOSS.withDescription("frontend/ViT image length mismatch").asRuntimeException();
            for (int j = 0; j < span.getLength(); j++) if (input.getTokenIds(span.getOffset() + j) != hashes.get(cursor++))
                throw Status.DATA_LOSS.withDescription("frontend/ViT image token identity mismatch").asRuntimeException();
            previousEnd = span.getOffset() + span.getLength();
        }
        return input;
    }

    int channelCount() { return channels.size(); }
    @Override public void close() {
        channels.values().forEach(ManagedChannel::shutdownNow);
        channels.clear();
    }
}
