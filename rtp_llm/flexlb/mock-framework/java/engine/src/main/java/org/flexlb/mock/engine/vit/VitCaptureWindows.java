package org.flexlb.mock.engine.vit;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Arms contiguous ViT capture windows and keeps their archives available for the current batch.
 * Hash responses record geometry in each open window so a request can finish after its window ends.
 */
final class VitCaptureWindows {
    private final String instance;
    private final VitNode.CaptureGeometry geometry;
    private final byte[] key;
    private final Map<Long, VitGeometryCapture> captures = new HashMap<>();
    private volatile List<VitGeometryCapture> active = List.of();
    private long run;
    private VitGeometryCapture latest;

    VitCaptureWindows(VitConfig config, String instance, byte[] key) {
        this.instance = instance;
        this.key = key.clone();
        geometry = new VitNode.CaptureGeometry(config.patch(), config.merge(), config.minPixels(), config.maxTokens());
    }

    synchronized VitNode.CaptureState arm(VitNode.CaptureArm request) throws Exception {
        VitGeometryCapture.checkArm(request, key);
        long now = System.currentTimeMillis();
        if (request.run() != run) {
            if (request.run() != request.start() || request.start() <= now
                    || active.stream().anyMatch(capture -> !capture.finished())) {
                throw new IllegalArgumentException("capture_window_unavailable");
            }
            captures.clear();
            latest = null;
            run = request.run();
        }
        var capture = captures.get(request.start());
        if (capture == null) {
            if (request.start() <= now || latest != null && latest.end() != request.start()) {
                throw new IllegalArgumentException("capture_window_unavailable");
            }
            capture = new VitGeometryCapture(request.start(), request.end(), key);
            captures.put(request.start(), capture);
            latest = capture;
            active = captures.values().stream().filter(item -> !item.finished()).toList();
        }
        if (capture.end() != request.end()) throw new IllegalArgumentException("capture_window_conflict");
        return new VitNode.CaptureState(capture.start(), capture.state(), instance, geometry);
    }

    VitNode.CaptureChunk read(VitNode.CaptureRead request) throws Exception {
        VitGeometryCapture capture;
        synchronized (this) {
            capture = captures.get(request.start());
        }
        if (capture == null) throw new IllegalArgumentException("capture_not_armed");
        return capture.read(request);
    }

    void record(VitGeometry.Image image) {
        for (var capture : active) {
            if (!capture.finished()) capture.record(image);
        }
    }
}
