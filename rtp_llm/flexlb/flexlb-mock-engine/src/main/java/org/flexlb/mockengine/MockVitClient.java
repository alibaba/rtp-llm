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
    private record Connection(String localName, ManagedChannel channel) {}
    private final Map<String, Connection> channels = new ConcurrentHashMap<>();
    java.util.function.BiFunction<String, Integer, String> localEndpoint = (host, port) -> null;

    private ManagedChannel channel(String host, int port) {
        String target = host + ":" + port;
        String localName = localEndpoint.apply(host, port);
        return channels.compute(target, (key, prior) -> {
            if (prior != null && java.util.Objects.equals(prior.localName(), localName)) return prior;
            if (prior != null) prior.channel().shutdownNow();
            ManagedChannel channel = localName == null
                    ? ManagedChannelBuilder.forAddress(host, port).usePlaintext()
                        .maxInboundMessageSize(64 << 20).disableRetry().build()
                    : io.grpc.inprocess.InProcessChannelBuilder.forName(localName).directExecutor().disableRetry().build();
            return new Connection(localName, channel);
        }).channel();
    }

    EngineRpcService.GenerateInputPB prepare(EngineRpcService.GenerateInputPB input, long budgetMs) {
        boolean typed = input.hasV41Inputs();
        if (typed) validateV41Layout(input);
        if (input.getMultimodalInputsCount() == 0 && (!typed || input.getV41Inputs().getImagesCount() == 0)) return input;
        var address = input.getGenerateConfig().getRoleAddrsList().stream()
                .filter(a -> a.getRoleStr().equals("VIT") || a.getRole() == EngineRpcService.RoleAddrPB.RoleType.VIT)
                .findFirst().orElseThrow(() -> Status.FAILED_PRECONDITION.withDescription("image request has no VIT role address").asRuntimeException());
        if (budgetMs <= 0) throw Status.DEADLINE_EXCEEDED.asRuntimeException();
        var channel = channel(address.getIp(), address.getGrpcPort());
        var request = EngineRpcService.MultimodalInputsPB.newBuilder().setRequestId(input.getRequestId())
                .addAllMultimodalInputs(input.getMultimodalInputsList()).setTimeoutMs(Math.min(budgetMs, 120000));
        if (typed) request.setV41Inputs(input.getV41Inputs());
        EngineRpcService.MultimodalOutputPB response;
        try {
            response = MultimodalRpcServiceGrpc.newBlockingStub(channel)
                    .withDeadlineAfter(Math.min(budgetMs, 120000), TimeUnit.MILLISECONDS).remoteMultimodalEmbedding(request.build());
        } catch (RuntimeException error) { throw downstreamFailure(error); }
        if (Context.current().isCancelled()) throw Status.CANCELLED.asRuntimeException();
        var features = response.getMultimodalEmbedding();
        long rows = response.getSplitSizeList().stream().mapToLong(Integer::longValue).sum();
        int segments = typed ? input.getV41Inputs().getImagesCount() : input.getMultimodalInputsCount();
        boolean bf16 = features.getDataType() == EngineRpcService.TensorPB.DataType.BF16;
        if (response.getSplitSizeCount() != segments
                || response.getSplitSizeList().stream().anyMatch(size -> size <= 0)
                || features.getShapeCount() != 2 || features.getShape(0) != rows
                || features.getShape(1) <= 0
                || (typed ? !bf16 : features.getDataType() != EngineRpcService.TensorPB.DataType.FP32)
                || !features.getFp32Data().isEmpty() || !features.getBf16Data().isEmpty()
                || !features.getInt32Data().isEmpty() || !features.getFp16Data().isEmpty())
            throw Status.DATA_LOSS.withDescription("invalid Mock ViT embedding shape or split sizes").asRuntimeException();
        try { Math.multiplyExact(Math.multiplyExact(rows, features.getShape(1)), bf16 ? 2 : 4); }
        catch (ArithmeticException overflow) {
            throw Status.DATA_LOSS.withDescription("ViT logical tensor size overflows").asRuntimeException();
        }
        if (typed) {
            for (int i = 0; i < segments; i++) if (response.getSplitSize(i) != input.getV41Inputs().getImages(i).getTypesCount())
                throw Status.DATA_LOSS.withDescription("V4.1 feature rows do not match canonical image span").asRuntimeException();
            return input; // Prepared frontend IDs are canonical; never replace them with feature hashes.
        }
        var hashes = new ArrayList<Integer>();
        var hashTensor = response.getMultimodalFeatureHash();
        var wire = hashTensor.getInt32Data().asReadOnlyByteBuffer().order(ByteOrder.LITTLE_ENDIAN);
        if (hashTensor.getDataType() != EngineRpcService.TensorPB.DataType.INT32
                || hashTensor.getShapeCount() != 1 || hashTensor.getShape(0) != rows
                || wire.remaining() != rows * 4)
            throw Status.DATA_LOSS.withDescription("invalid ViT hash metadata").asRuntimeException();
        while (wire.hasRemaining()) hashes.add(wire.getInt());
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

    /** MMGrpcTransport preserves named domain errors; only UNAVAILABLE/deadline become retryable RPC failures. */
    static RuntimeException downstreamFailure(Throwable error) {
        var status = Status.fromThrowable(error);
        long code = MockRpcErrors.downstream(error, 0).getErrorCode();
        if (code == 8211 && status.getCode() == Status.Code.RESOURCE_EXHAUSTED) code = 409;
        String message = status.getDescription() == null ? "" : status.getDescription();
        String[] names = {"MM_LONG_PROMPT_ERROR", "MM_WRONG_FORMAT_ERROR", "MM_PROCESS_ERROR",
                "MM_EMPTY_ENGINE_ERROR", "MM_NOT_SUPPORTED_ERROR", "MM_DOWNLOAD_FAILED", "MM_REMOTE_RPC_FAILED",
                "UNSAFE_INPUT_CONTENT", "UNSAFE_OUTPUT_CONTENT"};
        for (int i = 0; i < names.length; i++) if (message.startsWith("[" + names[i] + "]")) code = 901 + i;
        if (message.startsWith("[CONCURRENCY_LIMIT_ERROR]")) code = 409;
        if (message.startsWith("[GENERATE_TIMEOUT]")) code = 603;
        if (message.startsWith("[CANCELLED]")) code = 8100;
        if (code == 0) code = switch (status.getCode()) {
            case CANCELLED -> 8100;
            case RESOURCE_EXHAUSTED -> 409;
            case UNAVAILABLE, DEADLINE_EXCEEDED -> 907;
            default -> 514;
        };
        return MockRpcErrors.error(status, code, message);
    }

    static boolean retryable(Throwable error) {
        long code = MockRpcErrors.downstream(downstreamFailure(error), 0).getErrorCode();
        return code == 903 || code == 904 || code == 906 || code == 907;
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

    Map<String, Long> snapshot() {
        long local = channels.values().stream().filter(connection -> connection.localName() != null).count();
        return Map.of("local_channels", local, "remote_channels", channels.size() - local);
    }
    int channelCount() { return channels.size(); }
    @Override public void close() {
        channels.values().forEach(connection -> connection.channel().shutdownNow());
        channels.clear();
    }
}
