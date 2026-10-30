package org.flexlb.mock.multimodal;

import com.google.protobuf.ByteString;
import io.grpc.Status;
import java.nio.*;
import java.nio.charset.StandardCharsets;
import java.security.*;
import java.util.*;
import org.flexlb.engine.grpc.EngineRpcService.*;

/** Native media and Mock metadata contract, shared by V producers and P consumers. */
public final class MediaInput {
    private MediaInput() {}
    public static boolean present(GenerateInputPB input) {
        return input.getMultimodalInputsCount() > 0 || input.hasV41Inputs();
    }
    public static MultimodalInputsPB media(GenerateInputPB input) {
        var media=MultimodalInputsPB.newBuilder().setRequestId(input.getRequestId())
            .addAllMultimodalInputs(input.getMultimodalInputsList()).setTimeoutMs(input.getGenerateConfig().getTimeoutMs());
        if(input.hasV41Inputs()) media.setV41Inputs(input.getV41Inputs());
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
    public record Image(String key, int width, int height, int rows) {
        public long bytes(int hidden, int dtypeBytes) { return Math.multiplyExact(Math.multiplyExact((long) rows, hidden), dtypeBytes); }
    }
    public static ByteString hashes(String key, int rows) {
        byte[] digest;
        try { digest = MessageDigest.getInstance("SHA-256").digest(key.getBytes(StandardCharsets.UTF_8)); }
        catch (NoSuchAlgorithmException error) { throw new AssertionError(error); }
        long hash = ByteBuffer.wrap(digest).getLong();
        var output = ByteBuffer.allocate(Math.multiplyExact(rows, 4)).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < rows; i++) {
            hash ^= 0x9e3779b97f4a7c15L + i + (hash << 12) + (hash >> 32);
            output.putInt((int) (hash ^ (hash >>> 32)));
        }
        return ByteString.copyFrom(output.array());
    }
    /** Inputs already contain expanded token spans, as on the native wire. */
    public static GenerateInputPB apply(GenerateInputPB input, MultimodalOutputPB output) {
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
        return applyRows(input,output.getSplitSizeList(),hashes.getInt32Data());
    }
    public static GenerateInputPB applyHashes(GenerateInputPB input, MultimodalHashResponsePB metadata) {
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
        return applyRows(input,sizes,payload);
    }
    private static GenerateInputPB applyRows(GenerateInputPB input,List<Integer> sizes,ByteString payload) {
        var spans = input.getMultimodalTokenLayout().getSpansList();
        if (spans.size() != input.getMultimodalInputsCount() || sizes.size() != spans.size())
            throw invalid("Each image requires one expanded token span");
        var data = payload.asReadOnlyByteBuffer().order(ByteOrder.LITTLE_ENDIAN);
        var prepared = input.toBuilder(); int previousEnd = 0;
        for (int i = 0; i < spans.size(); i++) {
            var span = spans.get(i); int length = sizes.get(i);
            if (length < 1 || span.getLength() != length || span.getOffset() < previousEnd
                    || (long) span.getOffset() + length > input.getTokenIdsCount() || data.remaining() < (long) length * 4)
                throw invalid("Expanded image token span differs from V output");
            for (int n = 0; n < length; n++) prepared.setTokenIds(span.getOffset() + n, data.getInt());
            previousEnd = span.getOffset() + length;
        }
        if (data.hasRemaining()) throw Status.DATA_LOSS.withDescription("Excess vision hashes").asRuntimeException();
        return prepared.build();
    }
    public static String address(GenerateInputPB input) {
        var addresses = input.getGenerateConfig().getRoleAddrsList().stream().filter(a -> a.getRole() == RoleAddrPB.RoleType.VIT).toList();
        if (addresses.size() != 1 || addresses.getFirst().getIp().isBlank() || addresses.getFirst().getGrpcPort() < 1 || addresses.getFirst().getGrpcPort() > 65535)
            throw invalid("Exactly one valid VIT role address is required");
        var a = addresses.getFirst(); return (a.getIp().contains(":") ? "[" + a.getIp() + "]" : a.getIp()) + ":" + a.getGrpcPort();
    }
    public static RuntimeException invalid(String message) { return Status.INVALID_ARGUMENT.withDescription(message).asRuntimeException(); }
}
