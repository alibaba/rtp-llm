package org.flexlb.mock.runtime;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import java.nio.file.*;
import static org.junit.jupiter.api.Assertions.*;

class CacheNamespaceTest {
    @TempDir Path directory;
    @Test void seedComesFromTheSameModelNamespaceAsRealFrontend() throws Exception {
        var json = new ObjectMapper();
        Files.writeString(directory.resolve("config.json"), "{\"model_type\":\"deepseek_v41\",\"cache_key_hash_seed\":123}");
        var settings = json.createObjectNode().put("checkpoint_path", directory.toString());
        assertEquals(0x4453563431494631L, EngineConfig.cacheKeySeed(settings));
        Files.writeString(directory.resolve("config.json"), "{\"cache_key_hash_seed\":123}");
        assertEquals(123, EngineConfig.cacheKeySeed(settings));
    }
}
