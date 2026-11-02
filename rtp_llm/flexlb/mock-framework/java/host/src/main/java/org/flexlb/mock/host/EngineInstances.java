package org.flexlb.mock.host;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.*;
import org.flexlb.mock.engine.rtp.*;
import org.flexlb.mock.runtime.EngineInstance;
import org.flexlb.mock.runtime.EngineConfig;
import org.flexlb.mock.api.model.Types.Role;
import java.util.Map;
import org.flexlb.mock.vit.*;

/** Same instance assembly for the local host and a single-engine Pod. */
final class EngineInstances {
    private static final ObjectMapper JSON = new ObjectMapper();

    static EngineInstance start(Path configuration, Path output, int rpcPort, int controlPort) throws Exception {
        return start(EngineConfig.read(configuration), output, rpcPort, controlPort);
    }

    static EngineInstance startEnvironment(Map<String, String> env, Path output, int rpcPort, int controlPort) throws Exception {
        String role = env.getOrDefault("ROLE_TYPE", "").toLowerCase(java.util.Locale.ROOT);
        String id = env.get("MOCK_NODE_ID");
        if (id == null || !id.startsWith(role + "-")) throw new IllegalArgumentException("Missing/invalid MOCK_NODE_ID for ROLE_TYPE");
        if (role.equals("vit")) {
            var config = VitConfig.fromEnvironment(id, env);
            writeEffective(output, config.effective());
            return VitNode.start(config, rpcPort, controlPort);
        }
        if (!role.equals("prefill") && !role.equals("decode")) throw new IllegalArgumentException("Unknown ROLE_TYPE: " + role);
        var config = RtpConfig.fromEnvironment(id, Role.valueOf(role), env);
        writeEffective(output, config.effective());
        return RtpNode.start(config, rpcPort, controlPort);
    }

    private static EngineInstance start(com.fasterxml.jackson.databind.JsonNode root, Path output, int rpcPort, int controlPort) throws Exception {
        String role = root.path("component_role").asText();
        if (role.isBlank() || !root.path("id").asText().startsWith(role + "-"))
            throw new IllegalArgumentException("Instance identity differs from role");
        if (role.equals("vit")) {
            var config = VitConfig.load(root);
            writeEffective(output, config.effective());
            return VitNode.start(config, rpcPort, controlPort);
        }
        if (!role.equals("prefill") && !role.equals("decode"))
            throw new IllegalArgumentException("Unknown Engine role: " + role);
        // P/D select an LLM backend; V uses one shared implementation.
        var config = RtpConfig.load(root);
        writeEffective(output, config.effective());
        return RtpNode.start(config, rpcPort, controlPort);
    }

    private static void writeEffective(Path output, Object config) throws Exception {
        // A fresh instance must not overwrite retired artifacts.
        try (var stream = Files.newOutputStream(output.resolve("effective-config.json"), StandardOpenOption.CREATE_NEW)) {
            JSON.writeValue(stream, config);
        }
    }

    static void publish(Path path, Object value) throws Exception {
        var temporary = Files.createTempFile(path.toAbsolutePath().getParent(), ".publish-", ".json");
        try {
            JSON.writeValue(temporary.toFile(), value);
            Files.move(temporary, path, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
        } finally { Files.deleteIfExists(temporary); }
    }
}
