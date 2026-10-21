package org.flexlb.mockengine;

/** CPU equivalent of main's cpp/multimodal_processor/FeatureHash.h. */
final class MockFeatureHash {
    private MockFeatureHash() {}
    static long mix(long value) {
        value ^= value >>> 30; value *= 0xbf58476d1ce4e5b9L;
        value ^= value >>> 27; value *= 0x94d049bb133111ebL;
        return value ^ (value >>> 31);
    }
    static int row(byte[] bytes, int offset, int length) {
        long hash = mix(0x243f6a8885a308d3L ^ length);
        for (int start = 0, index = 0; start < length; start += 8, index++) {
            long word = 0;
            for (int i = 0; i < 8 && start + i < length; i++)
                word |= (bytes[offset + start + i] & 255L) << (8 * i);
            hash ^= mix(word ^ mix(0x9e3779b97f4a7c15L + index));
        }
        hash = mix(hash ^ 0x9e3779b97f4a7c15L);
        return (int) (hash ^ (hash >>> 32));
    }

    /** libstdc++ std::hash<string>, used by production Linux MultimodalInput::cache_key. */
    static long linuxString(byte[] bytes) {
        final long m = 0xc6a4a7935bd1e995L;
        long hash = 0xc70f6907L ^ (bytes.length * m);
        int offset = 0;
        for (; offset + 8 <= bytes.length; offset += 8) {
            long word = 0;
            for (int i = 0; i < 8; i++) word |= (bytes[offset + i] & 255L) << (8 * i);
            word *= m; word ^= word >>> 47; word *= m;
            hash ^= word; hash *= m;
        }
        if (offset < bytes.length) {
            long tail = 0;
            for (int i = 0; offset + i < bytes.length; i++) tail |= (bytes[offset + i] & 255L) << (8 * i);
            hash ^= tail; hash *= m;
        }
        hash ^= hash >>> 47; hash *= m; return hash ^ (hash >>> 47);
    }
}
