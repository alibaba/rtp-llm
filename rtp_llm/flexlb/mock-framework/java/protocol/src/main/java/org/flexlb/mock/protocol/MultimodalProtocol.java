package org.flexlb.mock.protocol;

import com.google.protobuf.ByteString;
import io.grpc.Status;
import java.nio.*;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.*;
import org.flexlb.engine.grpc.EngineRpcService.*;

/**
 * Shares image identity, feature hashes, and multimodal message conversion across Client, ViT, and Prefill.
 */
public final class MultimodalProtocol {
    public static final int IMAGE_TOKEN = -200;

    private MultimodalProtocol() {}

    public static boolean present(GenerateInputPB input) {
        return input.getMultimodalInputsCount() > 0;
    }

    public static String key(MultimodalInputPB input) {
        if (input.getMultimodalUrl().isBlank()) throw invalid("Image cache identity must be nonempty");
        var p = input.getMmPreprocessConfig();
        String crop = p.getCropPositionsList().stream().map(v -> String.format(Locale.ROOT,"%.6f",v)).collect(java.util.stream.Collectors.joining(":"));
        return Long.toUnsignedString(linuxString(input.getMultimodalUrl().getBytes(StandardCharsets.UTF_8)))+"_"+input.getMultimodalType()+"_"
            +p.getWidth()+"_"+p.getHeight()+"_"+p.getMinPixels()+"_"+p.getMaxPixels()+"_"+p.getFps()+"_"+p.getMinFrames()+"_"+p.getMaxFrames()+"_"+crop+"_"
            +(p.getMaxLongSidePixel()>0?p.getMaxLongSidePixel():-1);
    }

    public static RuntimeException invalid(String message) { return Status.INVALID_ARGUMENT.withDescription(message).asRuntimeException(); }

    public static Prepared text(GenerateInputPB input) { return new Prepared(input, List.of()); }

    /** Expands the seeds returned by the Mock ViT embedding RPC. */
    public static Prepared apply(GenerateInputPB input, MultimodalOutputPB output, int imageToken) {
        var tensor = output.getMultimodalEmbedding();
        if (output.getOutputRdmaSlotsCount() != 0 || !tensor.getFp32Data().isEmpty() || !tensor.getBf16Data().isEmpty()
                || !tensor.getFp16Data().isEmpty() || !tensor.getInt32Data().isEmpty()
                || tensor.getShapeCount() != 2 || tensor.getShape(0) < 1 || tensor.getShape(1) < 1
                || tensor.getDataType() != TensorPB.DataType.BF16)
            throw Status.DATA_LOSS.withDescription("Malformed Mock vision metadata").asRuntimeException();
        if (output.hasMultimodalPosId() || output.getMultimodalExtraInputCount() != 0)
            throw Status.DATA_LOSS.withDescription("Unexpected vision tensor payload").asRuntimeException();
        var sizes = output.getSplitSizeList();
        if (output.hasMultimodalFeatureHash() || sizes.size() != output.getMockFeatureSeedsCount()
                || sizes.size() != input.getMultimodalInputsCount()
                || sizes.stream().anyMatch(size -> size < 1)
                || sizes.stream().mapToLong(Integer::longValue).sum() != tensor.getShape(0))
            throw Status.DATA_LOSS.withDescription("Invalid Mock feature seeds").asRuntimeException();
        ByteString payload = ByteString.EMPTY;
        for (int i = 0; i < sizes.size(); i++)
            payload = payload.concat(expandFeatureHashes(output.getMockFeatureSeeds(i), sizes.get(i)));
        return applyRows(input, sizes, payload, imageToken);
    }

    public static ByteString expandFeatureHashes(long hash, int rows) {
        var output = ByteBuffer.allocate(Math.multiplyExact(rows, 4)).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < rows; i++) {
            hash ^= 0x9e3779b97f4a7c15L + i + (hash << 12) + (hash >> 32);
            output.putInt((int) (hash ^ (hash >>> 32)));
        }
        return ByteString.copyFrom(output.array());
    }

    public static PromptCacheKeys computePromptKeys(Prepared prepared, int blockSize, long seed, int keyStride) {
        validateSpans(prepared.input(), prepared.spans());
        return new PromptCacheKeys(blockSize, prepared.input().getTokenIdsList(),
            prepared.spans().stream().map(span -> new PromptCacheKeys.Span(
                span.getOffset(), span.getRowHashesList())).toList(), seed, keyStride);
    }

    public static long featureSeed(String key) {
        byte[] digest;
        try { digest = MessageDigest.getInstance("SHA-256").digest(key.getBytes(StandardCharsets.UTF_8)); }
        catch (NoSuchAlgorithmException error) { throw new AssertionError(error); }
        return ByteBuffer.wrap(digest).getLong();
    }

    static MultimodalInputsPB media(GenerateInputPB input) {
        var media=MultimodalInputsPB.newBuilder().setRequestId(input.getRequestId())
            .addAllMultimodalInputs(input.getMultimodalInputsList()).setTimeoutMs(input.getGenerateConfig().getTimeoutMs());
        return media.build();
    }

    // Linux libstdc++ string hash used by MultimodalInput::cache_key, not Java String.hashCode.
    static long linuxString(byte[] bytes) {
        final long m=0xc6a4a7935bd1e995L;
        long hash=0xc70f6907L ^ (bytes.length*m);int offset=0;
        for(;offset+8<=bytes.length;offset+=8) {
            long word=ByteBuffer.wrap(bytes,offset,8).order(ByteOrder.LITTLE_ENDIAN).getLong();
            word*=m;word^=word>>>47;word*=m;hash^=word;hash*=m;
        }
        if(offset<bytes.length) {
            long tail=0;for(int i=0;offset+i<bytes.length;i++) tail|=(bytes[offset+i]&255L)<<(8*i);
            hash^=tail;hash*=m;
        }
        hash^=hash>>>47;hash*=m;return hash^(hash>>>47);
    }

    static Prepared applyHashes(GenerateInputPB input, MultimodalHashResponsePB metadata, int imageToken) {
        var entries=new HashMap<String,MultimodalHashEntryPB>();
        for(var entry:metadata.getEntriesList()) {
            if(entries.putIfAbsent(entry.getKey(),entry)!=null) throw Status.DATA_LOSS.withDescription("Duplicate vision hash entry").asRuntimeException();
        }
        var sizes=new ArrayList<Integer>();ByteString payload=ByteString.EMPTY;
        for(var media:input.getMultimodalInputsList()) {
            var entry=entries.get(key(media));
            if(entry==null || !entry.getHashHit() || entry.getSplitSizeCount()!=1 || entry.getSplitSize(0)<1
                    || !entry.hasMockFeatureSeed() || !entry.getFeatureHashes().isEmpty())
                throw Status.DATA_LOSS.withDescription("Missing or invalid vision feature hashes").asRuntimeException();
            int rows = entry.getSplitSize(0);
            sizes.add(rows);
            payload = payload.concat(expandFeatureHashes(entry.getMockFeatureSeed(), rows));
        }
        return applyRows(input,sizes,payload,imageToken);
    }

    static String address(GenerateInputPB input) {
        var addresses = input.getGenerateConfig().getRoleAddrsList().stream().filter(a -> a.getRole() == RoleAddrPB.RoleType.VIT).toList();
        if (addresses.size() != 1 || addresses.getFirst().getIp().isBlank() || addresses.getFirst().getGrpcPort() < 1 || addresses.getFirst().getGrpcPort() > 65535)
            throw invalid("Exactly one valid VIT role address is required");
        var a = addresses.getFirst(); return (a.getIp().contains(":") ? "[" + a.getIp() + "]" : a.getIp()) + ":" + a.getGrpcPort();
    }

    private static Prepared applyRows(GenerateInputPB input, List<Integer> sizes, ByteString payload, int imageToken) {
        if (sizes.size() != input.getMultimodalInputsCount()) throw invalid("Image output count mismatch");
        var data = payload.asReadOnlyByteBuffer().order(ByteOrder.LITTLE_ENDIAN);
        var result = input.toBuilder().clearTokenIds().clearMultimodalInputs();
        var spans = new ArrayList<MultimodalCacheSpanPB>();
        int image = 0;
        for (int token : input.getTokenIdsList()) {
            if (token != imageToken) { result.addTokenIds(token); continue; }
            if (image >= sizes.size()) throw invalid("Too many image placeholders");
            int rows = sizes.get(image++);
            if (rows < 1 || data.remaining() < (long) rows * 4)
                throw invalid("Invalid image output length");
            var span = MultimodalCacheSpanPB.newBuilder().setOffset(result.getTokenIdsCount());
            for (int row = 0; row < rows; row++) {
                result.addTokenIds(imageToken);
                span.addRowHashes(data.getInt());
            }
            spans.add(span.build());
        }
        if (image != sizes.size() || data.hasRemaining()) throw invalid("Image placeholders and hashes differ");
        return new Prepared(result.build(), spans);
    }

    private static void validateSpans(GenerateInputPB input, List<MultimodalCacheSpanPB> spans) {
        long previous = 0;
        for (var span : spans) {
            long end = (long) span.getOffset() + span.getRowHashesCount();
            if (span.getOffset() < previous || span.getRowHashesCount() < 1 || end > input.getTokenIdsCount())
                throw invalid("Invalid multimodal cache span");
            previous = end;
        }
    }

    /**
     * Pairs the native generation input with resolved media cache spans.
     */
    public record Prepared(GenerateInputPB input, List<MultimodalCacheSpanPB> spans) {
        public Prepared { spans = List.copyOf(spans); }
    }
}
