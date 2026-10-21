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
        boolean typed = input.hasV41Inputs();
        if (typed) validateV41Layout(input);
        if (input.getMultimodalInputsCount() == 0 && (!typed || input.getV41Inputs().getImagesCount() == 0)) return input;
        var address = input.getGenerateConfig().getRoleAddrsList().stream()
                .filter(a -> a.getRoleStr().equals("VIT") || a.getRole() == EngineRpcService.RoleAddrPB.RoleType.VIT)
                .findFirst().orElseThrow(() -> Status.FAILED_PRECONDITION.withDescription("image request has no VIT role address").asRuntimeException());
        if (budgetMs <= 0) throw Status.DEADLINE_EXCEEDED.asRuntimeException();
        String target = address.getIp() + ":" + address.getGrpcPort();
        var channel = channels.computeIfAbsent(target, ignored -> ManagedChannelBuilder
                .forAddress(address.getIp(), address.getGrpcPort()).usePlaintext()
                .maxInboundMessageSize(64 << 20).disableRetry().build());
        var request = EngineRpcService.MultimodalInputsPB.newBuilder().setRequestId(input.getRequestId())
                .addAllMultimodalInputs(input.getMultimodalInputsList()).setTimeoutMs(Math.min(budgetMs, 120000));
        if (typed) request.setV41Inputs(input.getV41Inputs());
        var response = MultimodalRpcServiceGrpc.newBlockingStub(channel)
                .withDeadlineAfter(Math.min(budgetMs, 120000), TimeUnit.MILLISECONDS).remoteMultimodalEmbedding(request.build());
        if (Context.current().isCancelled()) throw Status.CANCELLED.asRuntimeException();
        var features = response.getMultimodalEmbedding();
        long rows = response.getSplitSizeList().stream().mapToLong(Integer::longValue).sum();
        int segments = typed ? input.getV41Inputs().getImagesCount() : input.getMultimodalInputsCount();
        boolean bf16 = features.getDataType() == EngineRpcService.TensorPB.DataType.BF16;
        var data = bf16 ? features.getBf16Data() : features.getFp32Data();
        if (response.getSplitSizeCount() != segments
                || response.getSplitSizeList().stream().anyMatch(size -> size <= 0)
                || features.getShapeCount() != 2 || features.getShape(0) != rows
                || features.getShape(1) <= 0
                || (typed ? !bf16 : features.getDataType() != EngineRpcService.TensorPB.DataType.FP32)
                || rows * features.getShape(1) * (bf16 ? 2 : 4) != data.size())
            throw Status.DATA_LOSS.withDescription("invalid Mock ViT embedding shape or split sizes").asRuntimeException();
        if (typed) {
            for (int i = 0; i < segments; i++) if (response.getSplitSize(i) != input.getV41Inputs().getImages(i).getTypesCount())
                throw Status.DATA_LOSS.withDescription("V4.1 feature rows do not match canonical image span").asRuntimeException();
            return input; // Prepared frontend IDs are canonical; never replace them with feature hashes.
        }
        byte[] tensor = data.toByteArray();
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

    private static void validateV41Layout(EngineRpcService.GenerateInputPB input) {
        var typed = input.getV41Inputs();
        int n = input.getTokenIdsCount();
        if (typed.getSchemaVersion() != 1 || input.getMultimodalInputsCount() != 0
                || typed.getTokenTypesCount() != n || typed.getImageMaskCount() != n)
            throw Status.INVALID_ARGUMENT.withDescription("V4.1 token metadata must match input vector").asRuntimeException();
        int[] expected = new int[n];
        java.util.Arrays.fill(expected, -1);
        int end = 0;
        for (var image : typed.getImagesList()) {
            if (image.getStart() < end || image.getTypesCount() == 0
                    || (long)image.getStart() + image.getTypesCount() > n)
                throw Status.INVALID_ARGUMENT.withDescription("invalid V4.1 image span").asRuntimeException();
            for (int j = 0; j < image.getTypesCount(); j++) {
                int type = image.getTypes(j);
                if (type < 0 || type > 3) throw Status.INVALID_ARGUMENT.withDescription("invalid V4.1 image type").asRuntimeException();
                expected[image.getStart() + j] = type;
            }
            end = image.getStart() + image.getTypesCount();
        }
        for (int i = 0; i < n; i++) if (typed.getTokenTypes(i) != expected[i] || typed.getImageMask(i) != (expected[i] != -1))
            throw Status.INVALID_ARGUMENT.withDescription("V4.1 image spans do not cover token metadata").asRuntimeException();
    }

    int channelCount() { return channels.size(); }
    @Override public void close() {
        channels.values().forEach(ManagedChannel::shutdownNow);
        channels.clear();
    }
}
