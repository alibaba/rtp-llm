package org.flexlb.mockengine;

import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

class LegacyImageCacheKeysTest {
    @Test void pageRrSamplesPhysicalImageHashesAfterIdentityMixing() {
        var request = EngineRpcService.GenerateInputPB.newBuilder();
        for (int i = 0; i < 1537; i++) request.addTokenIds(i);
        var image = request.getV41InputsBuilder().addImagesBuilder().setStart(600)
                .setNVitH(10).setNVitW(20).setContentSha256("ab".repeat(32)).setProcessorIdentity("cd".repeat(32));
        for (int i = 0; i < 200; i++) image.addTypes(1);
        var input = request.build();
        for (long seed : List.of(0L, 0x4453563431525031L)) {
            var physical = LegacyImageCacheKeys.fullBlocks(input, 128, seed);
            assertEquals(List.of(physical.get(3), physical.get(7), physical.get(11)),
                    LegacyImageCacheKeys.routeBlocks(input, 128, 512, seed));
            assertEquals(physical, LegacyImageCacheKeys.routeBlocks(input, 128, 128, seed));
            assertNotEquals(LegacyImageCacheKeys.fullBlocks(input, 512, seed),
                    LegacyImageCacheKeys.routeBlocks(input, 128, 512, seed));
        }
        assertThrows(IllegalArgumentException.class, () -> LegacyImageCacheKeys.routeBlocks(input, 128, 200, 0));
    }
    private EngineRpcService.GenerateInputPB.Builder input() {
        var input = EngineRpcService.GenerateInputPB.newBuilder();
        for (int i = 10; i < 16; i++) input.addTokenIds(i);
        input.getV41InputsBuilder().addImagesBuilder().setStart(1).setNVitH(3).setNVitW(3)
                .addTypes(0).addTypes(1).addTypes(2).addTypes(3)
                .setContentSha256("00000001".repeat(8)).setProcessorIdentity("ab".repeat(32));
        return input;
    }
    @Test void matchesReleaseCppImageIdentityAndLayoutSeed() {
        // Linux C++ HashUtil + CompleteTokenIds identity reference output.
        assertEquals(List.of(4321259559819448846L, -5298172462924781660L, 5540608416351323980L),
                LegacyImageCacheKeys.fullBlocks(input().build(), 2, 0));
        assertEquals(List.of(435955913959206881L, -3778471606194979373L, -6881559086064067850L),
                LegacyImageCacheKeys.fullBlocks(input().build(), 2, 0x4453563431525031L));
        var changed = input();
        changed.getV41InputsBuilder().getImagesBuilder(0).setContentSha256("01".repeat(32));
        assertNotEquals(LegacyImageCacheKeys.fullBlocks(input().build(), 2, 0),
                LegacyImageCacheKeys.fullBlocks(changed.build(), 2, 0));
    }
    @Test void reuseCannotCutPreparedImageOrConsumeLastPromptToken() {
        var request = input().build();
        assertEquals(0, LegacyImageCacheKeys.validHitBlocks(request, 2, 2));
        assertEquals(0, LegacyImageCacheKeys.validHitBlocks(request, 3, 2));
        assertEquals(0, LegacyImageCacheKeys.fullBlocks(request, 8, 0).size());
    }
}
