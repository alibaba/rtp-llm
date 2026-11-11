package org.flexlb.mock.engine.vit;

import io.grpc.Status;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.mock.protocol.MultimodalProtocol;

/**
 * V-owned image geometry, logical output size and simulated feature hashes.
 */
final class VitGeometry {
    /** The DSV4.1 grid planner, without tensor materialization. */
    static int rows(int width, int height, int patch, int downsample, int minPixels, int maxTokens) {
        if ((long) width * height < minPixels) {
            double scale = Math.sqrt((double) minPixels / ((long) width * height));
            width = (int) (width * scale); height = (int) (height * scale);
        }
        long resizedW = ((width + (long) patch - 1) / patch) * patch;
        long resizedH = ((height + (long) patch - 1) / patch) * patch;
        long llmW = (resizedW / patch + downsample - 1) / downsample;
        long llmH = (resizedH / patch + downsample - 1) / downsample;
        if ((long) llmH * (llmW + 1) + 2 > maxTokens) {
            double ratio = (double) height / width;
            double maxW = Math.sqrt((maxTokens - 2) / ratio + .25) - .5;
            double maxH = maxW * ratio;
            long cell = Math.multiplyExact((long) patch, downsample);
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
        if(input.getTimeoutMs()<0) throw MultimodalProtocol.invalid("Invalid media timeout");
        if(input.getMultimodalInputsCount()<1) throw MultimodalProtocol.invalid("Expected nonempty image inputs");
        for(var media:input.getMultimodalInputsList()) {
            if(media.getMultimodalType()!=1 || media.hasMultimodalTensor())
                throw Status.UNIMPLEMENTED.withDescription("V supports URL image inputs only").asRuntimeException();
            MultimodalProtocol.key(media);
            if(skipDownload) dimensions(media.getMmPreprocessConfig().getWidth(),media.getMmPreprocessConfig().getHeight());
        }
    }

    static void dimensions(int width,int height) {
        if(width<1 || height<1)
            throw MultimodalProtocol.invalid("Image width/height must be positive; skip-download inputs require explicit dimensions");
    }

    /**
     * Carries source and resized image dimensions and the resulting token count.
     */
    record Image(String key, int width, int height, int rows, String featureIdentity) {
        Image(String key, int width, int height, int rows) { this(key, width, height, rows, key); }
        public long bytes(int hidden, int dtypeBytes) { return Math.multiplyExact(Math.multiplyExact((long) rows, hidden), dtypeBytes); }
    }
}
