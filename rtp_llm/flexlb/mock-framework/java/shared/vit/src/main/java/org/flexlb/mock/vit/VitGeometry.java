package org.flexlb.mock.vit;

import io.grpc.Status;
import com.google.protobuf.ByteString;
import java.nio.*;
import java.nio.charset.StandardCharsets;
import java.security.*;
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

    /** The DSV4.1 grid planner, without pixel decode or tensor materialization. */
    static int rows(int width, int height, int patch, int downsample, int minPixels, int maxTokens) {
        if ((long) width * height < minPixels) {
            double scale = Math.sqrt((double) minPixels / ((long) width * height));
            width = (int) (width * scale); height = (int) (height * scale);
        }
        int resizedW = ((width + patch - 1) / patch) * patch;
        int resizedH = ((height + patch - 1) / patch) * patch;
        int llmW = (resizedW / patch + downsample - 1) / downsample;
        int llmH = (resizedH / patch + downsample - 1) / downsample;
        if ((long) llmH * (llmW + 1) + 2 > maxTokens) {
            double ratio = (double) height / width;
            double maxW = Math.sqrt((maxTokens - 2) / ratio + .25) - .5;
            double maxH = maxW * ratio;
            int cell = Math.multiplyExact(patch, downsample);
            if (maxW < 1) { resizedH = ((maxTokens - 2) / 2) * cell; resizedW = cell; }
            else if (maxH < 1) { resizedH = cell; resizedW = (maxTokens - 3) * cell; }
            else {
                double scale = Math.min(Math.floor(maxW) * cell / width, Math.floor(maxH) * cell / height);
                resizedH = (int) Math.floor(height * scale / patch) * patch;
                resizedW = (int) Math.floor(width * scale / patch) * patch;
            }
            llmW = (resizedW / patch + downsample - 1) / downsample;
            llmH = (resizedH / patch + downsample - 1) / downsample;
        }
        return Math.toIntExact((long) llmH * (llmW + 1) + 2);
    }
    static void validate(MultimodalInputsPB input, boolean skipDownload) {
        if(input.getTimeoutMs()<0 || input.getTimeoutMs()>3600000) throw MediaInput.invalid("Invalid media timeout");
        if(input.getMultimodalInputsCount()<1 || input.getMultimodalInputsCount()>256) throw MediaInput.invalid("Expected 1..256 images");
        for(var media:input.getMultimodalInputsList()) {
            if(media.getMultimodalType()!=1 || media.hasMultimodalTensor())
                throw Status.UNIMPLEMENTED.withDescription("V supports URL image inputs only").asRuntimeException();
            MediaInput.key(media);
            if(skipDownload) dimensions(media.getMmPreprocessConfig().getWidth(),media.getMmPreprocessConfig().getHeight());
        }
    }
    static void dimensions(int width,int height) {
        if(width<1 || height<1 || width>32768 || height>32768)
            throw MediaInput.invalid("Image width/height must be in 1..32768; skip-download inputs require explicit dimensions");
    }
}
