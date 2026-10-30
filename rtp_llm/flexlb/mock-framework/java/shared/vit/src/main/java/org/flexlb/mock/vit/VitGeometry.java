package org.flexlb.mock.vit;

import io.grpc.Status;
import com.google.protobuf.ByteString;
import java.nio.*;
import java.nio.charset.StandardCharsets;
import java.security.*;
import java.util.*;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.mock.multimodal.MediaInput;

/** V-owned image geometry, logical output size and simulated feature hashes. */
final class VitGeometry {
    record Image(String key, int width, int height, int rows) {
        public long bytes(int hidden, int dtypeBytes) { return Math.multiplyExact(Math.multiplyExact((long) rows, hidden), dtypeBytes); }
    }
    static ByteString hashes(String key, int rows) {
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

    static List<Image> images(MultimodalInputsPB input, int patch, int merge) {
        if (input.getTimeoutMs()<0 || input.getTimeoutMs()>3600000) throw MediaInput.invalid("V timeout must be in 0..3600000 ms");
        if (input.hasV41Inputs()) throw Status.UNIMPLEMENTED.withDescription("Prepared V4.1 patch layout needs its own image geometry model").asRuntimeException();
        if (input.getMultimodalInputsCount() < 1 || input.getMultimodalInputsCount() > 256)
            throw MediaInput.invalid("Expected 1..256 images");
        var result = new ArrayList<Image>();
        for (var media : input.getMultimodalInputsList()) {
            if (media.getMultimodalType() != 1) throw Status.UNIMPLEMENTED.withDescription("Only image media are modeled by V").asRuntimeException();
            if (media.hasMultimodalTensor()) throw Status.UNIMPLEMENTED.withDescription("Tensor media need an explicit geometry model").asRuntimeException();
            var shape = media.getMmPreprocessConfig();
            int width = shape.getWidth(), height = shape.getHeight();
            if (width < 1 || height < 1 || width > 32768 || height > 32768)
                throw MediaInput.invalid("Mock image inputs require explicit width/height in mm_preprocess_config, in 1..32768");
            long stride = (long) patch * merge;
            long rows = Math.multiplyExact((width + stride - 1) / stride, (height + stride - 1) / stride);
            if (rows < 1 || rows > 1048576) throw MediaInput.invalid("Image output exceeds row limit");
            result.add(new Image(MediaInput.key(media), width, height, (int) rows));
        }
        if (result.stream().mapToLong(Image::rows).sum() > 1048576) throw MediaInput.invalid("Request output exceeds row limit");
        return List.copyOf(result);
    }
}
