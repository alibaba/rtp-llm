package org.flexlb.mock.client;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import org.flexlb.mock.protocol.ConfigInput;
import org.flexlb.mock.protocol.JsonWire;
import org.flexlb.mock.protocol.MultimodalProtocol;

/**
 * Starts the runner-owned replay client, publishes its endpoints, and keeps it alive until shutdown.
 */
public final class ClientMain {
    private static final ObjectMapper JSON = new ObjectMapper();

    private ClientMain() {}

    public static void main(String[] args) throws Exception {
        if (args.length != 0) throw new IllegalArgumentException("Use runner environment");
        var config = JSON.readTree(Path.of(required("MOCK_CONFIG_FILE")).toFile());
        var settings = config.path("component").path("config");
        var allowed = Set.of("image_token_id", "cache_key_hash_seed", "seq_size_per_block", "prefill_cp_size");
        settings.fieldNames().forEachRemaining(name -> {
            if (!allowed.contains(name)) throw new IllegalArgumentException("Unknown Client setting: " + name);
        });
        var configuredImageToken = settings.get("image_token_id");
        if (configuredImageToken != null && (!configuredImageToken.isIntegralNumber()
                || !configuredImageToken.canConvertToInt() || configuredImageToken.intValue() < 0))
            throw new IllegalArgumentException("Client image_token_id must be nonnegative int32");
        int imageToken = settings.path("image_token_id").asInt(MultimodalProtocol.IMAGE_TOKEN);
        var configuredBlockSize = settings.get("seq_size_per_block");
        int physicalBlockSize = settings.path("seq_size_per_block").asInt(0);
        if (configuredBlockSize != null && (!configuredBlockSize.isIntegralNumber()
                || !configuredBlockSize.canConvertToInt() || physicalBlockSize < 1))
            throw new IllegalArgumentException("Client seq_size_per_block must be positive int32");
        var configuredCp = settings.get("prefill_cp_size");
        if (configuredCp != null && (!configuredCp.isIntegralNumber()
                || !configuredCp.canConvertToInt() || configuredCp.intValue() < 0))
            throw new IllegalArgumentException("Client prefill_cp_size must be nonnegative int32");
        int prefillCpSize = Math.max(1, settings.path("prefill_cp_size").asInt(1));
        if (prefillCpSize > 1 && physicalBlockSize == 0)
            throw new IllegalArgumentException("Client prefill_cp_size requires seq_size_per_block");
        int tokensPerRoutingKey = Math.multiplyExact(physicalBlockSize, prefillCpSize);
        long seed = ConfigInput.signedLong(settings, "cache_key_hash_seed", 0);

        Path directory = Path.of(required("MOCK_OUTPUT_DIR"));
        String id = required("MOCK_NODE_ID");
        Path topology = Path.of(required("MOCK_TOPOLOGY_FILE"));
        try (var client = new ClientNode(id, topology, directory, imageToken, physicalBlockSize, seed, prefillCpSize)) {
            var hook = new Thread(client::close, "client-shutdown");
            Runtime.getRuntime().addShutdownHook(hook);
            var effective = JSON.createObjectNode().put("id", id)
                .put("seq_size_per_block", physicalBlockSize).put("cache_key_hash_seed", seed)
                .put("prefill_cp_size", prefillCpSize).put("cache_key_block_size", tokensPerRoutingKey);
            if (configuredImageToken != null) effective.put("image_token_id", imageToken);
            Files.createDirectories(directory);
            var writer = JSON.writerWithDefaultPrettyPrinter();
            JsonWire.writeAtomically(directory.resolve("effective-config.json"), effective, writer, true);
            String url = client.url();
            JsonWire.writeAtomically(directory.resolve("endpoints.json"), Map.of("id", id, "role", "client",
                "control", url, "rpc", URI.create(url).getAuthority(), "metrics", url + "/metrics"), writer, true);
            client.publishReady(Path.of(required("MOCK_READY_FILE")));
            try {
                new CountDownLatch(1).await();
            } finally {
                try { Runtime.getRuntime().removeShutdownHook(hook); } catch (IllegalStateException ignored) { }
            }
        }
    }

    private static String required(String name) {
        String value = System.getenv(name);
        if (value == null || value.isBlank()) throw new IllegalArgumentException("Missing " + name);
        return value;
    }
}
