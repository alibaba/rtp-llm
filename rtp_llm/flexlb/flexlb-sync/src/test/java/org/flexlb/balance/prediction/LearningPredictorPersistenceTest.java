package org.flexlb.balance.prediction;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LearningPredictorPersistenceTest {

    private static final String STATE_MAGIC = "flexlb_learning_predictor";

    /** Nine representative doubles, including round-trip-sensitive values. */
    private static final double[] STATE_WEIGHTS = {
            Math.PI, -Math.E, 1.0e-15, 123456789.123456789, -0.0,
            Double.MAX_VALUE / 3, 1.0e300, -4.40538432604287, 0.08305932028650383 };

    @Test
    @DisplayName("save 后 load 完整往返 weights/generation/history 且浮点保真")
    void saveThenLoadRoundTripsFullState(@TempDir Path tempDir) throws IOException {
        Path stateFile = tempDir.resolve("state.json");
        LearningPredictorPersistence persistence =
                new LearningPredictorPersistence(stateFile, 100, 8);
        recordSamples(persistence, 3);

        persistence.save(STATE_WEIGHTS, 42L);

        LearningPredictorPersistence.LoadedState loaded =
                new LearningPredictorPersistence(stateFile, 100, 8).load();
        assertArrayEquals(STATE_WEIGHTS, loaded.weights(),
                "weights must survive the JSON round trip bit-exactly");
        assertEquals(42L, loaded.generation(), "generation must round trip");
        assertEquals(3, loaded.history().size(),
                "all recorded samples must round trip");
        assertFalse(loaded.refitOnStart(),
                "usable weights must not request a cold-start refit");
        assertEquals(300L, loaded.history().get(0).actualMs(),
                "sample labels must round trip");
        assertEquals(500L, loaded.history().get(0).features().items().get(0).seqLen(),
                "sample features must round trip");
    }

    @Test
    @DisplayName("load 文件缺失时返回冷启动空状态")
    void loadMissingFileReturnsColdStartEmpty(@TempDir Path tempDir) {
        LearningPredictorPersistence.LoadedState loaded =
                new LearningPredictorPersistence(tempDir.resolve("absent.json"), 10, 4).load();

        assertNull(loaded.weights(),
                "a missing state file is a normal cold start without weights");
        assertEquals(0L, loaded.generation(), "a cold start resets the generation");
        assertTrue(loaded.history().isEmpty(), "a cold start restores no history");
        assertFalse(loaded.refitOnStart(), "an empty cold start never requests a refit");
    }

    @Test
    @DisplayName("load magic 篡改时 ERROR 降级为冷启动")
    void loadTamperedMagicDegradesToColdStart(@TempDir Path tempDir) throws IOException {
        Path stateFile = tempDir.resolve("state.json");
        LearningPredictorPersistence persistence =
                new LearningPredictorPersistence(stateFile, 10, 4);
        recordSamples(persistence, 2);
        persistence.save(STATE_WEIGHTS, 7L);
        tamperStateFile(stateFile,
                "\"" + STATE_MAGIC + "\"", "\"tampered_magic\"");

        LearningPredictorPersistence.LoadedState loaded =
                new LearningPredictorPersistence(stateFile, 10, 4).load();
        assertNull(loaded.weights(), "a tampered magic must discard the parameters");
        assertTrue(loaded.history().isEmpty(),
                "a magic mismatch must also drop the retained history");
        assertFalse(loaded.refitOnStart(), "no history survives, so no refit is requested");
    }

    @Test
    @DisplayName("load stateVersion 不符时降级为冷启动")
    void loadUnknownStateVersionDegradesToColdStart(@TempDir Path tempDir) throws IOException {
        Path stateFile = tempDir.resolve("state.json");
        LearningPredictorPersistence persistence =
                new LearningPredictorPersistence(stateFile, 10, 4);
        recordSamples(persistence, 2);
        persistence.save(STATE_WEIGHTS, 7L);
        tamperStateFile(stateFile, "\"stateVersion\":1", "\"stateVersion\":2");

        LearningPredictorPersistence.LoadedState loaded =
                new LearningPredictorPersistence(stateFile, 10, 4).load();
        assertNull(loaded.weights(), "an unknown state version must discard the parameters");
        assertTrue(loaded.history().isEmpty(),
                "a version mismatch must also drop the retained history");
        assertFalse(loaded.refitOnStart(), "no history survives, so no refit is requested");
    }

    @Test
    @DisplayName("load paramCount 与 weights 长度不一致时丢参数、留历史并请求 refit")
    void loadMismatchedParamCountKeepsHistoryAndRequestsRefit(@TempDir Path tempDir)
            throws IOException {
        Path stateFile = tempDir.resolve("state.json");
        LearningPredictorPersistence persistence =
                new LearningPredictorPersistence(stateFile, 10, 4);
        recordSamples(persistence, 2);
        persistence.save(STATE_WEIGHTS, 9L);
        tamperStateFile(stateFile, "\"paramCount\":9", "\"paramCount\":8");

        LearningPredictorPersistence.LoadedState loaded =
                new LearningPredictorPersistence(stateFile, 10, 4).load();
        assertNull(loaded.weights(),
                "mismatched paramCount makes the saved parameters unusable");
        assertEquals(0L, loaded.generation(),
                "unusable parameters reset the generation to zero");
        assertEquals(2, loaded.history().size(),
                "the retained history must survive unusable parameters");
        assertTrue(loaded.refitOnStart(),
                "surviving history with dropped parameters must request a refit");
    }

    @Test
    @DisplayName("load weights 含非有限值时丢参数、留历史并请求 refit")
    void loadNonFiniteWeightsKeepsHistoryAndRequestsRefit(@TempDir Path tempDir)
            throws IOException {
        Path stateFile = tempDir.resolve("state.json");
        Files.writeString(stateFile, "{\"magic\":\"" + STATE_MAGIC + "\",\"stateVersion\":1,"
                + "\"paramCount\":2,\"generation\":9,\"weights\":[1.5,1e400],"
                + "\"history\":[{\"features\":{\"items\":[{\"seqLen\":500,\"hitCache\":200}]},"
                + "\"actualMs\":400}]}");

        LearningPredictorPersistence.LoadedState loaded =
                new LearningPredictorPersistence(stateFile, 10, 4).load();
        assertNull(loaded.weights(),
                "weights parsed to Infinity must be treated as unusable");
        assertEquals(0L, loaded.generation(),
                "unusable parameters reset the generation to zero");
        assertEquals(1, loaded.history().size(),
                "the retained history must survive unusable parameters");
        assertTrue(loaded.refitOnStart(),
                "surviving history with dropped parameters must request a refit");
    }

    @Test
    @DisplayName("load 文件截断或非法 JSON 时降级为空状态")
    void loadTruncatedOrMalformedFileDegradesToEmpty(@TempDir Path tempDir) throws IOException {
        Path truncated = tempDir.resolve("truncated.json");
        LearningPredictorPersistence persistence =
                new LearningPredictorPersistence(truncated, 10, 4);
        recordSamples(persistence, 2);
        persistence.save(STATE_WEIGHTS, 1L);
        String json = Files.readString(truncated);
        Files.writeString(truncated, json.substring(0, json.length() / 2));
        assertEmptyColdStart(new LearningPredictorPersistence(truncated, 10, 4).load(),
                "a truncated state file must degrade to a cold start");

        Path malformed = tempDir.resolve("malformed.json");
        Files.writeString(malformed, "{not valid json");
        assertEmptyColdStart(new LearningPredictorPersistence(malformed, 10, 4).load(),
                "a malformed state file must degrade to a cold start");

        Path empty = tempDir.resolve("empty.json");
        Files.writeString(empty, "");
        assertEmptyColdStart(new LearningPredictorPersistence(empty, 10, 4).load(),
                "an empty state file must degrade to a cold start");

        Path nonNumeric = tempDir.resolve("non-numeric.json");
        Files.writeString(nonNumeric, "{\"magic\":\"" + STATE_MAGIC + "\",\"stateVersion\":1,"
                + "\"paramCount\":1,\"generation\":1,\"weights\":[NaN],\"history\":[]}");
        assertEmptyColdStart(new LearningPredictorPersistence(nonNumeric, 10, 4).load(),
                "a NaN weight literal must fail parsing and degrade to a cold start");
    }

    @Test
    @DisplayName("load 历史超过 historyLimit 时只取末尾且顺序保持")
    void loadRetainsOnlyLatestHistoryUpToLimit(@TempDir Path tempDir) throws IOException {
        Path stateFile = tempDir.resolve("state.json");
        LearningPredictorPersistence persistence =
                new LearningPredictorPersistence(stateFile, 5, 1000);
        for (int i = 0; i < 8; i++) {
            persistence.recordSample(features(500L, 100L), i * 10L);
        }
        persistence.save(STATE_WEIGHTS, 1L);

        LearningPredictorPersistence.LoadedState loaded =
                new LearningPredictorPersistence(stateFile, 5, 1000).load();
        assertEquals(5, loaded.history().size(),
                "load must retain only the newest historyLimit samples");
        for (int i = 0; i < 5; i++) {
            assertEquals((3 + i) * 10L, loaded.history().get(i).actualMs(),
                    "retained history must keep chronological order, oldest first");
        }
        assertArrayEquals(STATE_WEIGHTS, loaded.weights(),
                "the weights must still restore when the history is trimmed");
    }

    @Test
    @DisplayName("recordSample 按 saveInterval 节流且无效样本不计数")
    void recordSampleThrottlesAtSaveInterval(@TempDir Path tempDir) {
        LearningPredictorPersistence persistence =
                new LearningPredictorPersistence(tempDir.resolve("state.json"), 100, 256);
        for (int i = 0; i < 255; i++) {
            assertFalse(persistence.recordSample(features(500L, 100L), 100L),
                    "samples before saveInterval must not request a save");
        }
        assertTrue(persistence.recordSample(features(500L, 100L), 100L),
                "the saveInterval-th new sample must request a save");
        for (int i = 0; i < 255; i++) {
            assertFalse(persistence.recordSample(features(500L, 100L), 100L),
                    "the throttle window must restart after each save");
        }
        assertTrue(persistence.recordSample(features(500L, 100L), 100L),
                "the second window must end at its own saveInterval-th sample");

        LearningPredictorPersistence everySample =
                new LearningPredictorPersistence(tempDir.resolve("every.json"), 100, 1);
        assertTrue(everySample.recordSample(features(500L, 100L), 100L),
                "saveInterval=1 must request a save on every sample");

        LearningPredictorPersistence guard =
                new LearningPredictorPersistence(tempDir.resolve("guard.json"), 100, 4);
        assertFalse(guard.recordSample(null, 100L),
                "null features must be ignored, not counted");
        assertFalse(guard.recordSample(features(500L, 100L), -1L),
                "negative labels must be ignored, not counted");
        for (int i = 0; i < 3; i++) {
            assertFalse(guard.recordSample(features(500L, 100L), 100L),
                    "invalid samples must not advance the throttle window");
        }
        assertTrue(guard.recordSample(features(500L, 100L), 100L),
                "only the fourth valid sample may request a save");
    }

    @Test
    @DisplayName("save 原子写完成后磁盘上没有 .tmp 残留")
    void saveLeavesNoTempFileBehind(@TempDir Path tempDir) throws IOException {
        Path stateFile = tempDir.resolve("state.json");
        LearningPredictorPersistence persistence =
                new LearningPredictorPersistence(stateFile, 10, 4);
        recordSamples(persistence, 2);

        persistence.save(STATE_WEIGHTS, 1L);

        assertTrue(Files.exists(stateFile), "the state file must exist after save");
        try (var files = Files.list(tempDir)) {
            List<Path> leftovers = files
                    .filter(path -> path.getFileName().toString().endsWith(".tmp"))
                    .toList();
            assertTrue(leftovers.isEmpty(),
                    "an atomic save must not leave .tmp files behind: " + leftovers);
        }
    }

    @Test
    @DisplayName("save 到不存在的目录时自动创建父目录")
    void saveCreatesMissingParentDirectories(@TempDir Path tempDir) throws IOException {
        Path stateFile = tempDir.resolve("a/b/c/state.json");
        LearningPredictorPersistence persistence =
                new LearningPredictorPersistence(stateFile, 10, 4);
        recordSamples(persistence, 1);

        persistence.save(STATE_WEIGHTS, 3L);

        assertTrue(Files.exists(stateFile),
                "save must create the missing parent directories");
        assertEquals(3L,
                new LearningPredictorPersistence(stateFile, 10, 4).load().generation(),
                "the state written into a fresh directory must load back");
    }

    @Test
    @DisplayName("save IO 失败时不抛异常且组件状态保持可用")
    void saveFailureIsLoggedNotThrown(@TempDir Path tempDir) throws IOException {
        Path blocked = tempDir.resolve("blocked");
        Files.createDirectory(blocked);
        LearningPredictorPersistence persistence =
                new LearningPredictorPersistence(blocked, 10, 4);
        assertDoesNotThrow(() -> persistence.save(STATE_WEIGHTS, 1L),
                "save failures must be logged, never propagated");
        assertDoesNotThrow(() -> persistence.save(STATE_WEIGHTS, 2L),
                "a failed save must leave the component usable for the next attempt");
        for (int i = 0; i < 3; i++) {
            assertFalse(persistence.recordSample(features(500L, 100L), 100L),
                    "recording must keep throttling after a failed save");
        }
        assertTrue(persistence.recordSample(features(500L, 100L), 100L),
                "the throttle window must still fire after a failed save");

        Path untouched = tempDir.resolve("untouched.json");
        LearningPredictorPersistence noop =
                new LearningPredictorPersistence(untouched, 10, 4);
        assertDoesNotThrow(() -> noop.save(null, 1L),
                "null weights must be ignored without an exception");
        assertDoesNotThrow(() -> noop.save(new double[0], 1L),
                "empty weights must be ignored without an exception");
        assertFalse(Files.exists(untouched),
                "empty weights must not produce a state file");
    }

    @Test
    @DisplayName("并发 recordSample 每个样本恰好入历史一次且无异常")
    void concurrentRecordSampleKeepsExactHistory(@TempDir Path tempDir) throws Exception {
        Path stateFile = tempDir.resolve("state.json");
        LearningPredictorPersistence persistence =
                new LearningPredictorPersistence(stateFile, 10000, 100000);
        int threadCount = 8;
        int samplesPerThread = 100;
        ExecutorService executor = Executors.newFixedThreadPool(threadCount);
        CountDownLatch start = new CountDownLatch(1);
        List<Future<?>> futures = new ArrayList<>();
        try {
            for (int t = 0; t < threadCount; t++) {
                final int threadIdx = t;
                futures.add(executor.submit(() -> {
                    start.await();
                    for (int i = 0; i < samplesPerThread; i++) {
                        persistence.recordSample(
                                features(1000L + threadIdx, threadIdx * 10L + i),
                                threadIdx * 1000L + i);
                    }
                    return null;
                }));
            }
            start.countDown();
            for (Future<?> future : futures) {
                future.get(10, TimeUnit.SECONDS);
            }
        } finally {
            executor.shutdownNow();
        }

        persistence.save(STATE_WEIGHTS, 1L);
        LearningPredictorPersistence.LoadedState loaded =
                new LearningPredictorPersistence(stateFile, 10000, 100000).load();
        assertEquals(threadCount * samplesPerThread, loaded.history().size(),
                "concurrent recording must keep every sample exactly once");
        long expectedSum = 0L;
        for (int t = 0; t < threadCount; t++) {
            for (int i = 0; i < samplesPerThread; i++) {
                expectedSum += t * 1000L + i;
            }
        }
        long actualSum = loaded.history().stream()
                .mapToLong(LearningPredictorPersistence.LearningSample::actualMs)
                .sum();
        assertEquals(expectedSum, actualSum,
                "no sample may be lost, duplicated or corrupted");
    }

    private static void assertEmptyColdStart(LearningPredictorPersistence.LoadedState loaded,
                                             String message) {
        assertNull(loaded.weights(), message);
        assertEquals(0L, loaded.generation(), message);
        assertTrue(loaded.history().isEmpty(), message);
        assertFalse(loaded.refitOnStart(), message);
    }

    private static PrefillBatchFeatures features(long seqLen, long hitCache) {
        return new PrefillBatchFeatures(
                List.of(new PrefillBatchFeatures.Item(seqLen, hitCache)));
    }

    private static void recordSamples(LearningPredictorPersistence persistence, int count) {
        for (int i = 0; i < count; i++) {
            persistence.recordSample(features(500L + i * 100L, 100L + i * 10L), 300L + i);
        }
    }

    private static void tamperStateFile(Path stateFile, String target, String replacement)
            throws IOException {
        String json = Files.readString(stateFile);
        assertTrue(json.contains(target),
                "the saved state must contain the fragment to tamper: " + target);
        Files.writeString(stateFile, json.replace(target, replacement));
    }
}
