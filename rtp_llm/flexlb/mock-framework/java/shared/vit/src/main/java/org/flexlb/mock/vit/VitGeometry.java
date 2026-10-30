package org.flexlb.mock.vit;

import io.grpc.Status;
import java.util.*;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.mock.multimodal.MediaInput;

/** Explicit image geometry model; media support restrictions belong only to V. */
final class VitGeometry {
    static List<MediaInput.Image> images(MultimodalInputsPB input, int patch, int merge) {
        if (input.getTimeoutMs()<0 || input.getTimeoutMs()>3600000) throw MediaInput.invalid("V timeout must be in 0..3600000 ms");
        if (input.hasV41Inputs()) throw Status.UNIMPLEMENTED.withDescription("Prepared V4.1 patch layout needs its own image geometry model").asRuntimeException();
        if (input.getMultimodalInputsCount() < 1 || input.getMultimodalInputsCount() > 256)
            throw MediaInput.invalid("Expected 1..256 images");
        var result = new ArrayList<MediaInput.Image>();
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
            result.add(new MediaInput.Image(MediaInput.key(media), width, height, (int) rows));
        }
        if (result.stream().mapToLong(MediaInput.Image::rows).sum() > 1048576) throw MediaInput.invalid("Request output exceeds row limit");
        return List.copyOf(result);
    }
}
