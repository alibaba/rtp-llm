package org.flexlb.mock.runtime;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.net.URI;
import java.net.http.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.security.*;
import java.time.Duration;
import java.util.HexFormat;
import java.util.Map;

/** Resolve CHECKPOINT_PATH through the same read-only C2 Fuse mount as real RTP. */
final class ModelPath {
    static Path resolve(String source, Map<String, String> environment) throws IOException {
        if (!source.contains("://")) return Path.of(source);
        if (source.startsWith("file://")) return Path.of(URI.create(source));
        String port = environment.getOrDefault("HIPPO_FUSE_PORT", "");
        String host = port.isBlank() ? "127.0.0.1" : environment.get("HIPPO_SLAVE_IP");
        String prefix = port.isBlank() ? "/mnt/fuse" : environment.get("HIPPO_APP_WORKDIR");
        if (host == null || prefix == null)
            throw new IOException("Host Fuse requires HIPPO_SLAVE_IP and HIPPO_APP_WORKDIR");
        Path directory = Path.of(prefix);
        if (!port.isBlank()) directory = directory.resolve("fuse");
        directory = directory.resolve(mountName(source));
        var json = new ObjectMapper();
        var payload = json.createObjectNode().put("uri", source).put("mountDir", directory.toString())
            .put("rwMode", "RWMODE_RO").put("enableMntRef", false);
        URI endpoint;
        try {
            int number = port.isBlank() ? 28006 : Integer.parseInt(port);
            if (number < 1 || number > 65535) throw new IllegalArgumentException();
            endpoint = new URI("http", null, host, number, "/FuseService/mount", null, null);
        } catch (Exception error) { throw new IOException("Invalid platform Fuse endpoint"); }
        try (var client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build()) {
            var request = HttpRequest.newBuilder(endpoint).timeout(Duration.ofSeconds(60))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(json.writeValueAsString(payload))).build();
            var response = client.send(request, HttpResponse.BodyHandlers.ofString());
            // Never include the remote URI, response body or its credentials in an error.
            if (response.statusCode() != 200)
                throw new IOException("Model Fuse mount returned HTTP " + response.statusCode());
            com.fasterxml.jackson.databind.JsonNode result;
            try { result = json.readTree(response.body()); }
            catch (com.fasterxml.jackson.core.JsonProcessingException error) {
                throw new IOException("Invalid model Fuse mount response");
            }
            if (result == null || !result.path("errorCode").isIntegralNumber()
                    || result.path("errorCode").longValue() != 0)
                throw new IOException("Model Fuse mount failed");
            return directory;
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            throw new IOException("Model Fuse mount interrupted");
        }
    }

    private static String mountName(String source) {
        try {
            return HexFormat.of().formatHex(MessageDigest.getInstance("MD5")
                .digest(source.getBytes(StandardCharsets.UTF_8)));
        } catch (NoSuchAlgorithmException error) { throw new AssertionError(error); }
    }

    private ModelPath() {}
}
