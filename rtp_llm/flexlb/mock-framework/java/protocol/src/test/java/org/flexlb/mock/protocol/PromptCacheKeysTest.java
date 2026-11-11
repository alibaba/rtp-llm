package org.flexlb.mock.protocol;

import static org.junit.jupiter.api.Assertions.*;
import java.util.*;
import org.junit.jupiter.api.Test;

/**
 * Checks cache-key generation for text and media inputs.
 */
class PromptCacheKeysTest {
    @Test void decodeHandoffPreservesOnlyCompletePromptKeys() throws Exception {
        for (int length : List.of(1, 7, 8, 9, 127, 128, 129)) {
            var tokens = Collections.nCopies(length, 19);
            var media = List.of(new PromptCacheKeys.Span(0, Collections.nCopies(length, 23)));
            var original = new PromptCacheKeys(8, tokens, media, 41);
            var wire = org.flexlb.engine.grpc.EngineRpcService.MockDecodePromptPB.parseFrom(original.decodePrompt().toByteArray());
            var decoded = PromptCacheKeys.fromDecodePrompt(wire, 8);
            assertEquals(length, decoded.inputLength());
            assertEquals(original.keys(), decoded.keys());
            assertEquals(length / 8, decoded.keys().size());
            for (int stride : List.of(1, 2, 4)) {
                var logical = original.withStride(stride);
                assertEquals(8 * stride, logical.blockSize());
                assertEquals(new PromptCacheKeys(8, tokens, media, 41, stride).keys(), logical.keys());
            }
        }
    }

    @Test void promptKeysAreImmutable() {
        var tokens = new ArrayList<>(List.of(1, 2, 3));
        var sequence = new PromptCacheKeys(2, tokens, List.of(), Long.MIN_VALUE);
        var expected = sequence.keys(); tokens.set(0, 99);
        assertSame(expected, sequence.keys());
        assertThrows(UnsupportedOperationException.class, () -> sequence.keys().add(1L));
        assertEquals(sequence, new PromptCacheKeys(2, List.of(1, 2, 3), List.of(), Long.MIN_VALUE));
    }
}
