package org.flexlb.mock.runtime;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpServer;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.HexFormat;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

class ModelPathTest {
    @TempDir Path directory;

    @Test void modelMetadataUsesTheSuppliedConfiguration() throws Exception {
        var json = new ObjectMapper();
        assertTrue(EngineConfig.model(json.createObjectNode(), Map.of()).isEmpty());
        var config = json.createObjectNode().put("checkpoint_path", directory.toString());
        json.writeValue(directory.resolve("config.json").toFile(), Map.of("cache_key_hash_seed", 17));
        assertEquals(17, EngineConfig.model(config, Map.of()).path("cache_key_hash_seed").longValue());
    }

    @Test void localPathsNeedNoFuseService() throws Exception {
        assertEquals(directory, ModelPath.resolve(directory.toString(), Map.of()));
        assertEquals(directory, ModelPath.resolve(directory.toUri().toString(), Map.of()));
    }

    @Test void remotePathsUseTheNativeReadOnlyMountContract() throws Exception {
        var request = new AtomicReference<String>();
        var server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/FuseService/mount", exchange -> {
            request.set(new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8));
            byte[] response = "{\"errorCode\":0}".getBytes(StandardCharsets.UTF_8);
            exchange.sendResponseHeaders(200, response.length);
            try (var output = exchange.getResponseBody()) { output.write(response); }
        });
        server.start();
        try {
            String source = "oss://bucket/model?OSS_ACCESS_KEY=test-secret";
            var env = Map.of("HIPPO_FUSE_PORT", Integer.toString(server.getAddress().getPort()),
                "HIPPO_SLAVE_IP", "127.0.0.1", "HIPPO_APP_WORKDIR", directory.toString());
            var mounted = ModelPath.resolve(source, env);
            String name = HexFormat.of().formatHex(MessageDigest.getInstance("MD5")
                .digest(source.getBytes(StandardCharsets.UTF_8)));
            assertEquals(directory.resolve("fuse").resolve(name), mounted);
            var payload = new ObjectMapper().readTree(request.get());
            assertEquals(source, payload.path("uri").asText());
            assertEquals(mounted.toString(), payload.path("mountDir").asText());
            assertEquals("RWMODE_RO", payload.path("rwMode").asText());
            assertFalse(payload.path("enableMntRef").asBoolean());
        } finally { server.stop(0); }
    }

    @Test void incompleteHostFuseConfigurationFailsWithoutExposingTheRemoteUri() {
        var error = assertThrows(java.io.IOException.class, () -> ModelPath.resolve(
            "oss://bucket/model?OSS_ACCESS_KEY=test-secret", Map.of("HIPPO_FUSE_PORT", "28006")));
        assertFalse(error.toString().contains("test-secret"));
    }
}
