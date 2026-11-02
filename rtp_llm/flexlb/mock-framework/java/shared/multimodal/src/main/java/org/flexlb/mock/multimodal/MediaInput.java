package org.flexlb.mock.multimodal;

import com.google.protobuf.ByteString;
import io.grpc.Status;
import java.nio.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import org.flexlb.engine.grpc.EngineRpcService.*;

/** Native media and Mock metadata contract, shared by V producers and P consumers. */
public final class MediaInput {
    private MediaInput() {}
    public static boolean present(GenerateInputPB input) {
        return input.getMultimodalInputsCount() > 0;
    }
    public static MultimodalInputsPB media(GenerateInputPB input) {
        var media=MultimodalInputsPB.newBuilder().setRequestId(input.getRequestId())
            .addAllMultimodalInputs(input.getMultimodalInputsList()).setTimeoutMs(input.getGenerateConfig().getTimeoutMs());
        return media.build();
    }
    public static String key(MultimodalInputPB input) {
        if (input.getMultimodalUrl().isBlank() || input.getMultimodalUrl().length() > 4096) throw invalid("Image cache identity must be nonempty");
        var p = input.getMmPreprocessConfig();
        String crop = p.getCropPositionsList().stream().map(v -> String.format(Locale.ROOT,"%.6f",v)).collect(java.util.stream.Collectors.joining(":"));
        return Long.toUnsignedString(linuxString(input.getMultimodalUrl().getBytes(StandardCharsets.UTF_8)))+"_"+input.getMultimodalType()+"_"
            +p.getWidth()+"_"+p.getHeight()+"_"+p.getMinPixels()+"_"+p.getMaxPixels()+"_"+p.getFps()+"_"+p.getMinFrames()+"_"+p.getMaxFrames()+"_"+crop+"_"
            +(p.getMaxLongSidePixel()>0?p.getMaxLongSidePixel():-1);
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
    public static final int IMAGE_TOKEN = -200;
    public record Prepared(GenerateInputPB input, List<MultimodalCacheSpanPB> spans) {
        public Prepared { spans = List.copyOf(spans); }
    }
    public static Prepared text(GenerateInputPB input) { return new Prepared(input, List.of()); }

    /** Inputs already contain expanded token spans, as on the native wire. */
    public static Prepared apply(GenerateInputPB input, MultimodalOutputPB output, int imageToken) {
        var tensor = output.getMultimodalEmbedding(); var hashes = output.getMultimodalFeatureHash();
        if (output.getOutputRdmaSlotsCount() != 0 || !tensor.getFp32Data().isEmpty() || !tensor.getBf16Data().isEmpty()
                || !tensor.getFp16Data().isEmpty() || !tensor.getInt32Data().isEmpty()
                || tensor.getShapeCount() != 2 || tensor.getShape(0) < 1 || tensor.getShape(0) > 1048576 || tensor.getShape(1) < 1 || tensor.getShape(1) > 65536
                || tensor.getDataType() != TensorPB.DataType.BF16 || hashes.getDataType() != TensorPB.DataType.INT32
                || hashes.getShapeCount() != 1 || hashes.getShape(0) != tensor.getShape(0)
                || hashes.getInt32Data().size() != tensor.getShape(0) * 4)
            throw Status.DATA_LOSS.withDescription("Malformed Mock vision metadata").asRuntimeException();
        if (output.hasMultimodalPosId() || output.getMultimodalExtraInputCount() != 0 || !hashes.getFp32Data().isEmpty()
                || !hashes.getBf16Data().isEmpty() || !hashes.getFp16Data().isEmpty())
            throw Status.DATA_LOSS.withDescription("Unexpected vision tensor payload").asRuntimeException();
        return applyRows(input,output.getSplitSizeList(),hashes.getInt32Data(),imageToken);
    }
    public static Prepared applyHashes(GenerateInputPB input, MultimodalHashResponsePB metadata, int imageToken) {
        var entries=new HashMap<String,MultimodalHashEntryPB>();
        for(var entry:metadata.getEntriesList()) {
            if(entries.putIfAbsent(entry.getKey(),entry)!=null) throw Status.DATA_LOSS.withDescription("Duplicate vision hash entry").asRuntimeException();
        }
        var sizes=new ArrayList<Integer>();ByteString payload=ByteString.EMPTY;
        for(var media:input.getMultimodalInputsList()) {
            var entry=entries.get(key(media));
            if(entry==null || !entry.getHashHit() || entry.getSplitSizeCount()!=1 || entry.getSplitSize(0)<1
                    || entry.getSplitSize(0)>1048576 || entry.getFeatureHashes().size()!=(long)entry.getSplitSize(0)*4)
                throw Status.DATA_LOSS.withDescription("Missing or invalid vision feature hashes").asRuntimeException();
            sizes.add(entry.getSplitSize(0));payload=payload.concat(entry.getFeatureHashes());
            if(payload.size()>4*1048576) throw Status.RESOURCE_EXHAUSTED.withDescription("Vision hashes exceed request limit").asRuntimeException();
        }
        return applyRows(input,sizes,payload,imageToken);
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
            if (rows < 1 || (long) result.getTokenIdsCount() + rows > 1048576 || data.remaining() < (long) rows * 4)
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

    /** Mirrors CompleteTokenIds::multimodalCacheIdentity and the native rolling block hash. */
    public static List<Long> cacheKeys(Prepared prepared, int blockSize) {
        if (blockSize < 1) throw invalid("Invalid cache block size");
        validateSpans(prepared.input(), prepared.spans());
        var keys = new ArrayList<Long>(); long hash = 0;
        int length = prepared.input().getTokenIdsCount();
        for (int start = 0; start + blockSize <= length; start += blockSize) {
            for (int i = start; i < start + blockSize; i++) hash = roll(hash, prepared.input().getTokenIds(i));
            for (var span : prepared.spans()) {
                int begin = Math.max(start, span.getOffset());
                int end = Math.min(start + blockSize, span.getOffset() + span.getRowHashesCount());
                if (begin >= end) continue;
                for (int value : new int[]{-42, begin-start, span.getRowHashesCount(), begin-span.getOffset(), end-begin}) hash = roll(hash,value);
                for (int i = begin; i < end; i++) hash = roll(hash, span.getRowHashes(i-span.getOffset()));
            }
            keys.add(hash);
        }
        return List.copyOf(keys);
    }
    public static void validateSpans(GenerateInputPB input, List<MultimodalCacheSpanPB> spans) {
        long previous = 0;
        for (var span : spans) {
            long end = (long) span.getOffset() + span.getRowHashesCount();
            if (span.getOffset() < previous || span.getRowHashesCount() < 1 || end > input.getTokenIdsCount())
                throw invalid("Invalid multimodal cache span");
            previous = end;
        }
    }
    private static long roll(long hash, long value) { return hash ^ (value + 0x9e3779b97f4a7c15L + (hash << 12) + (hash >> 32)); }
    public static String address(GenerateInputPB input) {
        var addresses = input.getGenerateConfig().getRoleAddrsList().stream().filter(a -> a.getRole() == RoleAddrPB.RoleType.VIT).toList();
        if (addresses.size() != 1 || addresses.getFirst().getIp().isBlank() || addresses.getFirst().getGrpcPort() < 1 || addresses.getFirst().getGrpcPort() > 65535)
            throw invalid("Exactly one valid VIT role address is required");
        var a = addresses.getFirst(); return (a.getIp().contains(":") ? "[" + a.getIp() + "]" : a.getIp()) + ":" + a.getGrpcPort();
    }
    public static RuntimeException invalid(String message) { return Status.INVALID_ARGUMENT.withDescription(message).asRuntimeException(); }
}
