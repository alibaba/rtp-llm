package org.flexlb.mockengine;

import org.junit.jupiter.api.Test;
import java.nio.charset.StandardCharsets;
import static org.junit.jupiter.api.Assertions.assertEquals;

class MockFeatureHashTest {
    @Test void matchesMainCppCpuHashIncludingPartialWords() {
        // Produced by main FeatureHash.h on Linux, not the Java implementation.
        byte[] bytes = new byte[70];
        for (int i = 0; i < bytes.length; i++) bytes[i] = (byte)(i * 31 + 17);
        int[] lengths = {0, 1, 7, 8, 9, 32, 65};
        int[] expected = {-783942732, -368761820, -11532792, -1814894221,
                -940810509, 1686162198, -2108506664};
        for (int i = 0; i < lengths.length; i++) assertEquals(expected[i], MockFeatureHash.row(bytes, 0, lengths[i]));
    }
    @Test void matchesLinuxLibstdcppStringIdentity() {
        String[] input = {"", "https://example.com/image.png", "中文url"};
        String[] expected = {"6142509188972423790", "2000644620635833837", "18006779371473338865"};
        for (int i = 0; i < input.length; i++) assertEquals(expected[i],
                Long.toUnsignedString(MockFeatureHash.linuxString(input[i].getBytes(StandardCharsets.UTF_8))));
    }
}
