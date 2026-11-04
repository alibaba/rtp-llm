package org.flexlb.mock.engine;

import com.fasterxml.jackson.databind.JsonNode;
import java.nio.file.*;
import org.flexlb.mock.engine.rtp.*;
import org.flexlb.mock.runtime.EngineInstance;
import org.flexlb.mock.runtime.EngineConfig;
import org.flexlb.mock.api.model.Types.Role;
import java.util.Map;
import org.flexlb.mock.vit.*;

/** Same instance assembly for the local host and a single-engine Pod. */
final class EngineInstances {

    record Started(EngineInstance node, JsonNode effective) {}

    static Started start(Path configuration, int rpcPort, int controlPort) throws Exception {
        return start(EngineConfig.read(configuration), rpcPort, controlPort);
    }

    static Started startEnvironment(Map<String, String> env, int rpcPort, int controlPort) throws Exception {
        String role = env.getOrDefault("ROLE_TYPE", "").toLowerCase(java.util.Locale.ROOT);
        String id = env.getOrDefault("MOCK_NODE_ID", role + "-" + org.flexlb.mock.runtime.EngineAddress.local().advertise());
        if (id == null || !id.startsWith(role + "-")) throw new IllegalArgumentException("Missing/invalid MOCK_NODE_ID for ROLE_TYPE");
        if (role.equals("vit")) {
            var config = VitConfig.fromEnvironment(id, env);
            return new Started(VitNode.start(config, rpcPort, controlPort), config.effective());
        }
        if (!role.equals("prefill") && !role.equals("decode")) throw new IllegalArgumentException("Unknown ROLE_TYPE: " + role);
        var config = RtpConfig.fromEnvironment(id, Role.valueOf(role), env);
        return new Started(RtpNode.start(config, rpcPort, controlPort), config.effective());
    }

    private static Started start(JsonNode root, int rpcPort, int controlPort) throws Exception {
        String role = root.path("component_role").asText();
        if (role.isBlank() || !root.path("id").asText().startsWith(role + "-"))
            throw new IllegalArgumentException("Instance identity differs from role");
        if (role.equals("vit")) {
            var config = VitConfig.load(root);
            return new Started(VitNode.start(config, rpcPort, controlPort), config.effective());
        }
        if (!role.equals("prefill") && !role.equals("decode"))
            throw new IllegalArgumentException("Unknown Engine role: " + role);
        // P/D select an LLM backend; V uses one shared implementation.
        var config = RtpConfig.load(root);
        return new Started(RtpNode.start(config, rpcPort, controlPort), config.effective());
    }

}
