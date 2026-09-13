package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DecisionPolicyConfig;
import org.flexlb.config.DispatcherConfig;
import org.junit.jupiter.api.Test;
import java.nio.file.Files;
import java.nio.file.Path;
import static org.junit.jupiter.api.Assertions.*;

class WhaleGlmProfileTest {
    @Test void deployedDocumentPassesProductionStrictParser() throws Exception {
        Path file = Path.of("../tools/whale_mock/glm53-calibration.json");
        var profile = new ObjectMapper().readTree(Files.readString(file));
        var config = ConfigService.parse(profile.get("master_config").toString());
        assertEquals(DecisionPolicyConfig.Type.SINGLE, config.getScheduler().getDecision().getType());
        assertEquals(DispatcherConfig.Type.BATCH, config.getDispatcher().getType());
        assertEquals(64L, config.getRouter().getRoles().getDecode().getAvailability().getMaxEngineRequests());
        assertNotNull(config.getRouter().getRoles().getPrefill().getCacheAffinity());
    }
}
