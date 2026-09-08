package org.flexlb.constraint;

import org.flexlb.constraint.ConstraintTreeModels.BuildRequest;
import org.junit.jupiter.api.Test;

import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import static org.junit.jupiter.api.Assertions.*;

class ConstraintTreeSidMappingTest {
    static ConstraintTreeSidMapping mapping(Map<String, Integer> tokens) {
        var digest = ConstraintTreeSidMapping.newDigest();
        ConstraintTreeSidMapping.update(digest, "rtp-sid-mapping-v1\n217216\n1699\n151645\n");
        new TreeMap<>(tokens).forEach((symbol, id) -> ConstraintTreeSidMapping.update(digest, symbol + "\t" + id + "\n"));
        return new ConstraintTreeSidMapping(HexFormat.of().formatHex(digest.digest()), 217216, 1699, 151645, tokens);
    }

    static BuildRequest request(long version, String... sids) {
        return new BuildRequest(version, "gul_item", null, null, null, null, List.of(sids));
    }

    @Test
    void mapsTwoLevelSymbolsUsingActualIdsAndPreservesLeadingZeros() {
        var mapping = mapping(Map.of("C1", 169967, "C2", 216546, "C030", 215835)).validated();
        var converted = mapping.convert(request(1, "C1C2", "C030C2", "C1C030"));
        assertArrayEquals(new int[]{169967, 216546}, converted.rqTokenIds().get(0));
        assertArrayEquals(new int[]{215835, 216546}, converted.rqTokenIds().get(1));
        assertArrayEquals(new int[]{169967, 215835}, converted.rqTokenIds().get(2));
        try (var builder = new ConstraintTreeBuilder()) {
            var decoded = ConstraintTreeCsrCodec.decode(ConstraintTreeCsrCodec.encode(builder.build(converted),
                    mapping.fingerprint(), ConstraintTreeRequestIdentity.fingerprint(request(1, "C1C2", "C030C2", "C1C030"))));
            assertEquals(mapping.fingerprint(), decoded.mappingFingerprint());
            assertEquals(3, decoded.sidCount());
        }
    }

    @Test
    void rejectsUnknownMalformedAndMismatchedMappings() {
        var good = mapping(Map.of("C1", 17)).validated();
        for (String sid : List.of("C2", "C1_C1", " C1", "C1bad", "C1C", "C01", "C1", "C1C1C1")) {
            assertThrows(IllegalArgumentException.class, () -> good.convert(request(1, sid)));
        }
        assertThrows(IllegalArgumentException.class, () ->
                new ConstraintTreeSidMapping("a".repeat(64), 217216, 1699, 151645, Map.of("C1", 17)).validated());
        assertThrows(IllegalArgumentException.class, () -> mapping(Map.of("C1", 17, "C2", 17)).validated());
    }

    @Test
    void contentIdentityIsOrderIndependentAndMetadataSensitive() {
        String fingerprint = ConstraintTreeRequestIdentity.fingerprint(request(1, "C1C2", "C3"));
        assertEquals(fingerprint, ConstraintTreeRequestIdentity.fingerprint(request(1, "C3", "C1C2", "C3")));
        assertNotEquals(fingerprint, ConstraintTreeRequestIdentity.fingerprint(request(1, "C1C3", "C3")));
        var otherModel = new BuildRequest(1, "other_model", null, null, null, null, List.of("C1C2", "C3"));
        assertNotEquals(fingerprint, ConstraintTreeRequestIdentity.fingerprint(otherModel));
    }
}
