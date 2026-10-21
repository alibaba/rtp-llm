package org.flexlb.mockengine;

import io.grpc.Context;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.Status;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.MultimodalRpcServiceGrpc;

import java.util.ArrayList;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

/** Release protocol only: metadata_only field 2 and a per-image response envelope. */
final class LegacyMockVitClient implements AutoCloseable {
    private final Map<String, ManagedChannel> channels = new ConcurrentHashMap<>();
    private final int imageTokenId;
    private final java.util.concurrent.ThreadPoolExecutor executor = new java.util.concurrent.ThreadPoolExecutor(
            4, 4, 0, TimeUnit.MILLISECONDS, new java.util.concurrent.ArrayBlockingQueue<>(64), task -> {
                Thread thread = new Thread(task, "mock-legacy-vit-fetch"); thread.setDaemon(true); return thread;
            }, new java.util.concurrent.ThreadPoolExecutor.AbortPolicy());
    static boolean hasImages(EngineRpcService.GenerateInputPB input) {
        return input.getMultimodalInputsCount() > 0 || input.hasV41Inputs();
    }
    void execute(Runnable task, io.grpc.stub.StreamObserver<?> observer) {
        Context context = Context.current();
        try {
            executor.execute(context.wrap(() -> {
                if (context.isCancelled()) { observer.onError(Status.CANCELLED.asRuntimeException()); return; }
                try { task.run(); } catch (RuntimeException error) { observer.onError(error); }
            }));
        } catch (java.util.concurrent.RejectedExecutionException error) {
            observer.onError(Status.RESOURCE_EXHAUSTED.withDescription("Mock P ViT fetch queue full").asRuntimeException());
        }
    }
    LegacyMockVitClient() { this(Integer.parseInt(System.getenv().getOrDefault("MOCK_VIT_IMAGE_TOKEN_ID", "-1"))); }
    LegacyMockVitClient(int imageTokenId) { this.imageTokenId = imageTokenId; }

    EngineRpcService.GenerateInputPB prepare(EngineRpcService.GenerateInputPB input) {
        boolean packed = input.hasV41Inputs();
        if (!packed && input.getMultimodalInputsCount() == 0) return input;
        var vit = input.getGenerateConfig().getRoleAddrsList().stream()
                .filter(a -> a.getRoleStr().equals("VIT") || a.getRole() == EngineRpcService.RoleAddrPB.RoleType.VIT)
                .findFirst().orElseThrow(() -> Status.FAILED_PRECONDITION.withDescription("image input has no selected VIT").asRuntimeException());
        var request = EngineRpcService.MultimodalInputsPB.newBuilder().addAllMultimodalInputs(input.getMultimodalInputsList());
        if (packed) request.setV41Inputs(input.getV41Inputs());
        String key = vit.getIp() + ":" + vit.getGrpcPort();
        var channel = channels.computeIfAbsent(key, ignored -> ManagedChannelBuilder.forAddress(vit.getIp(), vit.getGrpcPort())
                .usePlaintext().disableRetry().maxInboundMessageSize(64 << 20).build());
        long budget = Context.current().getDeadline() == null ? 120000 : Context.current().getDeadline().timeRemaining(TimeUnit.MILLISECONDS);
        if (budget <= 0) throw Status.DEADLINE_EXCEEDED.asRuntimeException();
        var result = MultimodalRpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(budget, TimeUnit.MILLISECONDS)
                .remoteMultimodalEmbedding(request.build());
        int expected = packed ? input.getV41Inputs().getImagesCount() : input.getMultimodalInputsCount();
        if (expected == 0 || result.getMultimodalOutputsCount() != expected)
            throw Status.DATA_LOSS.withDescription("legacy ViT image count mismatch").asRuntimeException();
        var segments = new ArrayList<java.util.List<Integer>>();
        for (int i = 0; i < expected; i++) {
            var output = result.getMultimodalOutputs(i);
            if (output.hasOutputRdma()) throw Status.UNIMPLEMENTED.withDescription("Mock P uses inline legacy ViT features").asRuntimeException();
            var tensor = output.getMultimodalEmbedding();
            byte[] bytes;
            int width;
            switch (tensor.getDataType()) {
                case BF16 -> { bytes = tensor.getBf16Data().toByteArray(); width = 2; }
                case FP32 -> { bytes = tensor.getFp32Data().toByteArray(); width = 4; }
                default -> throw Status.UNIMPLEMENTED.withDescription("Mock P supports BF16/FP32 ViT features").asRuntimeException();
            }
            if (tensor.getShapeCount() != 2 || tensor.getShape(0) <= 0 || tensor.getShape(1) <= 0
                    || tensor.getShape(0) * tensor.getShape(1) * width != bytes.length)
                throw Status.DATA_LOSS.withDescription("invalid legacy ViT feature shape").asRuntimeException();
            int rows = Math.toIntExact(tensor.getShape(0)), stride = Math.toIntExact(tensor.getShape(1) * width);
            if (packed && rows != input.getV41Inputs().getImages(i).getTypesCount())
                throw Status.DATA_LOSS.withDescription("V4.1 canonical image length does not match ViT rows").asRuntimeException();
            var tokens = new ArrayList<Integer>(rows);
            for (int row = 0; row < rows; row++) tokens.add(MockFeatureHash.row(bytes, row * stride, stride));
            segments.add(tokens);
        }
        if (packed) {
            if (input.getV41Inputs().getTokenTypesCount() != input.getTokenIdsCount()
                    || input.getV41Inputs().getImageMaskCount() != input.getTokenIdsCount())
                throw Status.INVALID_ARGUMENT.withDescription("V4.1 token metadata length mismatch").asRuntimeException();
            return input; // Canonical V4.1 IDs belong to frontend preprocessing, not feature hashes.
        }
        if (imageTokenId < 0) throw Status.FAILED_PRECONDITION
                .withDescription("set MOCK_VIT_IMAGE_TOKEN_ID to the release model image placeholder ID").asRuntimeException();
        var tokens = new ArrayList<Integer>();
        int image = 0;
        for (int id : input.getTokenIdsList()) {
            if (id != imageTokenId) tokens.add(id);
            else {
                if (image == segments.size()) throw Status.INVALID_ARGUMENT.withDescription("too many image placeholders").asRuntimeException();
                tokens.addAll(segments.get(image++));
            }
        }
        if (image != segments.size()) throw Status.INVALID_ARGUMENT.withDescription("image placeholder count mismatch").asRuntimeException();
        return input.toBuilder().clearTokenIds().addAllTokenIds(tokens).build();
    }
    @Override public void close() { executor.shutdownNow(); channels.values().forEach(ManagedChannel::shutdownNow); channels.clear(); }
}
