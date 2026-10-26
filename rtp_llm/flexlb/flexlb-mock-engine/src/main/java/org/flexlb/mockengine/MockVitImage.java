package org.flexlb.mockengine;

import com.google.protobuf.ByteString;
import org.flexlb.engine.grpc.EngineRpcService;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.time.Duration;
import java.util.Base64;
import java.util.Locale;

/** Qwen2/2.5-VL image geometry; forward values are deterministic CPU simulation. */
final class MockVitImage {
    record Result(EngineRpcService.TensorPB embedding, EngineRpcService.TensorPB hashes, int rows) {
        // Shape describes logical tensor storage; embedding payload is deliberately absent.
        long embeddingBytes() { return Math.multiplyExact(Math.multiplyExact(embedding.getShape(0), embedding.getShape(1)),
                embedding.getDataType() == EngineRpcService.TensorPB.DataType.BF16 ? 2 : 4); }
        long hashBytes() { return hashes.getInt32Data().size(); }
    }
    record Prepared(int rows, int hiddenSize, long seed, double downloadMs) {}
    private MockVitImage() {}

    static String key(EngineRpcService.MultimodalInputPB image) {
        var p = image.getMmPreprocessConfig();
        String crop = p.getCropPositionsList().stream().map(x -> String.format(Locale.ROOT, "%.6f", x))
                .collect(java.util.stream.Collectors.joining(":"));
        return Long.toUnsignedString(MockFeatureHash.linuxString(image.getMultimodalUrl().getBytes(StandardCharsets.UTF_8)))
                + "_" + image.getMultimodalType() + "_" + p.getWidth() + "_" + p.getHeight()
                + "_" + p.getMinPixels() + "_" + p.getMaxPixels() + "_" + p.getFps()
                + "_" + p.getMinFrames() + "_" + p.getMaxFrames() + "_" + crop + "_"
                + (p.getMaxLongSidePixel() > 0 ? p.getMaxLongSidePixel() : -1);
    }

    static int[] resize(int height, int width, int minPixels, int maxPixels) {
        final int factor = 28;
        if (height <= 0 || width <= 0 || (double)Math.max(height, width) / Math.min(height, width) > 200)
            throw new IllegalArgumentException("invalid Qwen2-VL image dimensions");
        int h = Math.max(factor, (int)Math.rint((double)height / factor) * factor);
        int w = Math.max(factor, (int)Math.rint((double)width / factor) * factor);
        if ((long)h * w > maxPixels) {
            double beta = Math.sqrt((double)height * width / maxPixels);
            h = (int)Math.floor(height / beta / factor) * factor;
            w = (int)Math.floor(width / beta / factor) * factor;
        } else if ((long)h * w < minPixels) {
            double beta = Math.sqrt((double)minPixels / ((long)height * width));
            h = (int)Math.ceil(height * beta / factor) * factor;
            w = (int)Math.ceil(width * beta / factor) * factor;
        }
        if (h <= 0 || w <= 0) throw new IllegalArgumentException("resize produced an empty image");
        return new int[]{h, w};
    }

    static Result compute(EngineRpcService.MultimodalInputPB input, int hiddenSize,
                          int maxInputBytes, long maxOutputBytes, long timeoutMs) throws Exception {
        return forward(prepare(input, hiddenSize, maxInputBytes, maxOutputBytes, timeoutMs, java.util.Map.of()));
    }

    static Prepared prepare(EngineRpcService.MultimodalInputPB input, int hiddenSize,
                            int maxInputBytes, long maxOutputBytes, long timeoutMs,
                            java.util.Map<String, String> headers) throws Exception {
        return prepare(input, hiddenSize, maxInputBytes, maxOutputBytes, timeoutMs, headers, null);
    }

    static Prepared prepare(EngineRpcService.MultimodalInputPB input, int hiddenSize,
                            int maxInputBytes, long maxOutputBytes, long timeoutMs,
                            java.util.Map<String, String> headers, MockVitMediaCache mediaCache) throws Exception {
        long downloadStarted = System.nanoTime();
        if (input.getMultimodalType() != 1) throw new IllegalArgumentException("Mock ViT currently supports IMAGE only");
        if (input.getMultimodalTensor().getSerializedSize() != 0)
            throw new IllegalArgumentException("Mock ViT requires URL images, not preprocessed tensors");
        String url = input.getMultimodalUrl();
        byte[] raw = mediaCache == null ? null : mediaCache.get(url);
        if (raw == null) {
            if (url.startsWith("data:")) {
                int comma = url.indexOf(',');
                if (comma < 0 || !url.substring(0, comma).endsWith(";base64"))
                    throw new IllegalArgumentException("image data URL must be base64");
                if ((long)(url.length() - comma - 1) * 3 / 4 > maxInputBytes)
                    throw new IllegalArgumentException("image exceeds byte limit");
                try { raw = Base64.getDecoder().decode(url.substring(comma + 1)); }
                catch (IllegalArgumentException error) { throw MockRpcErrors.error(io.grpc.Status.INTERNAL, 906, "Multimodal download failed: invalid base64"); }
            } else {
                URI uri = URI.create(url);
                if (!"http".equals(uri.getScheme()) && !"https".equals(uri.getScheme()))
                    throw new IllegalArgumentException("image URL must use http, https or data");
                var client = HttpClient.newBuilder().connectTimeout(Duration.ofMillis(timeoutMs))
                        .followRedirects(HttpClient.Redirect.NORMAL).build();
                var request = HttpRequest.newBuilder(uri).timeout(Duration.ofMillis(timeoutMs));
                headers.forEach(request::header);
                long started = System.nanoTime();
                try {
                    var response = client.send(request.GET().build(), HttpResponse.BodyHandlers.ofInputStream());
                    // ofInputStream completes at the headers. Real's downloader also
                    // bounds body reads; closing the stream releases a stalled origin.
                    try (var reads = java.util.concurrent.Executors.newVirtualThreadPerTaskExecutor();
                         var body = response.body()) {
                        if (response.statusCode() != 200) throw MockRpcErrors.error(io.grpc.Status.INTERNAL, 906,
                                "Multimodal download failed: HTTP " + response.statusCode());
                        long remaining = timeoutMs - java.util.concurrent.TimeUnit.NANOSECONDS
                                .toMillis(System.nanoTime() - started);
                        var read = reads.submit(() -> body.readNBytes(maxInputBytes + 1));
                        try { raw = read.get(Math.max(1, remaining), java.util.concurrent.TimeUnit.MILLISECONDS); }
                        catch (java.util.concurrent.TimeoutException error) {
                            read.cancel(true);
                            throw MockRpcErrors.error(io.grpc.Status.INTERNAL, 906,
                                    "[MM_DOWNLOAD_FAILED] Download multimodal file timed out");
                        } catch (InterruptedException error) {
                            read.cancel(true);
                            throw error;
                        } catch (java.util.concurrent.ExecutionException error) {
                            if (error.getCause() instanceof java.io.IOException failure) throw failure;
                            throw error;
                        }
                    }
                } catch (java.io.IOException error) {
                    throw MockRpcErrors.error(io.grpc.Status.INTERNAL, 906, "Multimodal download failed: " + error.getMessage());
                }
            }
            if (raw.length > maxInputBytes) throw new IllegalArgumentException("image exceeds byte limit");
            if (mediaCache != null) mediaCache.put(url, raw);
        }
        if (raw.length > maxInputBytes) throw new IllegalArgumentException("image exceeds byte limit");
        double downloadMs = (System.nanoTime() - downloadStarted) / 1e6;
        BufferedImage image;
        try (var stream = ImageIO.createImageInputStream(new ByteArrayInputStream(raw))) {
            var readers = ImageIO.getImageReaders(stream);
            if (!readers.hasNext()) throw new IllegalArgumentException("unsupported or invalid image");
            var reader = readers.next();
            try {
                reader.setInput(stream, true, true);
                // Mock-specific decoded-pixel cap, checked before allocating the raster.
                if ((long)reader.getWidth(0) * reader.getHeight(0) > (32L << 20))
                    throw new IllegalArgumentException("image exceeds Mock decoded-pixel limit");
                image = reader.read(0);
            } finally { reader.dispose(); }
        }
        var p = input.getMmPreprocessConfig();
        int minPixels = p.getMinPixels() > 0 ? p.getMinPixels() : 4 * 28 * 28;
        int maxPixels = p.getMaxPixels() > 0 ? p.getMaxPixels() : 16384 * 28 * 28;
        if (minPixels > maxPixels) throw new IllegalArgumentException("min_pixels exceeds max_pixels");
        var geometry = resize(p.getHeight() > 0 ? p.getHeight() : image.getHeight(),
                p.getWidth() > 0 ? p.getWidth() : image.getWidth(), minPixels, maxPixels);
        int rows = Math.multiplyExact(geometry[0] / 28, geometry[1] / 28);
        long bytes = Math.multiplyExact((long)rows, Math.multiplyExact(hiddenSize, 4));
        if (bytes > maxOutputBytes || bytes > Integer.MAX_VALUE)
            throw new IllegalArgumentException("embedding exceeds output byte limit");
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        for (int y = 0; y < image.getHeight(); y++) for (int x = 0; x < image.getWidth(); x++) {
            int rgb = image.getRGB(x, y);
            digest.update((byte)(rgb >>> 16)); digest.update((byte)(rgb >>> 8)); digest.update((byte)rgb);
        }
        String cacheKey = key(input);
        digest.update(cacheKey.substring(cacheKey.indexOf('_')).getBytes(StandardCharsets.UTF_8));
        digest.update(ByteBuffer.allocate(8).putInt(geometry[0]).putInt(geometry[1]).array());
        byte[] seed = digest.digest();
        return new Prepared(rows, hiddenSize, ByteBuffer.wrap(seed).getLong(), downloadMs);
    }

    static Result forward(Prepared image) {
        int rows = image.rows(), hiddenSize = image.hiddenSize();
        long state = image.seed();
        ByteBuffer features = ByteBuffer.allocate(Math.multiplyExact(hiddenSize, 4)).order(ByteOrder.LITTLE_ENDIAN);
        ByteBuffer hashes = ByteBuffer.allocate(rows * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (int row = 0; row < rows; row++) {
            features.clear();
            for (int col = 0; col < hiddenSize; col++) {
                state = MockFeatureHash.mix(state + 0x9e3779b97f4a7c15L);
                features.putFloat(((state >>> 40) / (float)(1 << 24)) * 2 - 1);
            }
            hashes.putInt(MockFeatureHash.row(features.array(), 0, hiddenSize * 4));
        }
        return new Result(EngineRpcService.TensorPB.newBuilder().setDataType(EngineRpcService.TensorPB.DataType.FP32)
                .addShape(rows).addShape(hiddenSize).build(),
                EngineRpcService.TensorPB.newBuilder().setDataType(EngineRpcService.TensorPB.DataType.INT32)
                        .addShape(rows).setInt32Data(ByteString.copyFrom(hashes.array())).build(), rows);
    }
}
