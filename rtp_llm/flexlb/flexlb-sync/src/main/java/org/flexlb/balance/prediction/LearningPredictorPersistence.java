package org.flexlb.balance.prediction;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;

/**
 * Rolling learning-sample history plus atomic JSON state-file persistence for
 * {@link LearningPredictor}.
 *
 * <p>Semantics ported from the navi_sched agent history resource:
 * <ul>
 *   <li>a bounded rolling history of learning samples (default 2000);</li>
 *   <li>throttled persistence — one state-file update per {@code saveInterval}
 *       new samples (default 256), written through a temp file plus an atomic
 *       move so readers never observe a torn file;</li>
 *   <li>a validating startup load — magic, state version, parameter count and
 *       finite weights must all match. A missing file is a normal cold start;
 *       a corrupted file logs one ERROR and degrades to cold start; parameters
 *       that fail validation are dropped with one WARN while the retained
 *       history survives and requests a cold-start refit.</li>
 * </ul>
 */
public final class LearningPredictorPersistence {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");
    private static final String STATE_MAGIC = "flexlb_learning_predictor";
    private static final int STATE_VERSION = 1;
    private static final long MAX_STATE_FILE_BYTES = 64L * 1024 * 1024;
    private static final ObjectMapper MAPPER = new ObjectMapper();

    /** One retained learning sample: the learn() input features and its actual label. */
    public record LearningSample(PrefillBatchFeatures features, long actualMs) {
    }

    /**
     * Startup load outcome. {@code weights == null} means the saved parameters
     * are unusable, so the caller either starts from the initial parameters or
     * refits {@code history}.
     */
    public record LoadedState(double[] weights, long generation,
                              List<LearningSample> history, boolean refitOnStart) {

        static LoadedState empty() {
            return new LoadedState(null, 0L, List.of(), false);
        }
    }

    /** JSON state-file layout; the record component names are the on-disk contract. */
    private record StateFile(String magic, int stateVersion, int paramCount,
                             long generation, double[] weights,
                             List<LearningSample> history) {
    }

    private final Path stateFile;
    private final int historyLimit;
    private final int saveInterval;
    private final ArrayDeque<LearningSample> history = new ArrayDeque<>();
    private int samplesSinceSave;

    public LearningPredictorPersistence(Path stateFile, int historyLimit, int saveInterval) {
        this.stateFile = stateFile;
        this.historyLimit = historyLimit;
        this.saveInterval = saveInterval;
    }

    /**
     * Load the state file. Missing file: normal cold start. Structurally
     * invalid file: one ERROR, then cold start. Valid structure with unusable
     * parameters: one WARN, then the parameters are dropped, the latest
     * {@code historyLimit} samples retained and {@code refitOnStart} raised
     * when any survived.
     */
    public synchronized LoadedState load() {
        if (!Files.exists(stateFile)) {
            return LoadedState.empty();
        }
        StateFile state;
        try {
            long sizeBytes = Files.size(stateFile);
            if (sizeBytes <= 0 || sizeBytes > MAX_STATE_FILE_BYTES) {
                throw new IOException("unexpected state file size: " + sizeBytes);
            }
            state = MAPPER.readValue(stateFile.toFile(), StateFile.class);
        } catch (IOException | RuntimeException error) {
            logger.error("learning predictor state file corrupted, falling back to cold start: "
                    + "file={}, error={}", stateFile, error.toString());
            return LoadedState.empty();
        }
        if (!STATE_MAGIC.equals(state.magic()) || state.stateVersion() != STATE_VERSION) {
            logger.error("learning predictor state file magic/version mismatch, "
                    + "falling back to cold start: file={}", stateFile);
            return LoadedState.empty();
        }
        List<LearningSample> retained = retainLatest(state.history());
        history.clear();
        history.addAll(retained);
        double[] weights = usableWeights(state);
        if (weights == null) {
            logger.warn("learning predictor saved parameters unusable, dropping them and "
                    + "retaining {} history samples for a cold-start refit: file={}",
                    retained.size(), stateFile);
        }
        return new LoadedState(weights, weights == null ? 0L : state.generation(),
                retained, weights == null && !retained.isEmpty());
    }

    /**
     * Append one learning sample to the rolling history, evicting the oldest
     * entry beyond {@code historyLimit}.
     *
     * @return true once {@code saveInterval} new samples have accumulated; the
     *         caller should then persist the current model state
     */
    public synchronized boolean recordSample(PrefillBatchFeatures features, long actualMs) {
        if (features == null || actualMs < 0) {
            return false;
        }
        history.addLast(new LearningSample(features, actualMs));
        while (history.size() > historyLimit) {
            history.removeFirst();
        }
        samplesSinceSave++;
        if (samplesSinceSave >= saveInterval) {
            samplesSinceSave = 0;
            return true;
        }
        return false;
    }

    /**
     * Atomically write the model state plus the current rolling history.
     * Write failures never propagate; they are logged and retried at the next
     * throttled save.
     */
    public void save(double[] weights, long generation) {
        if (weights == null || weights.length == 0) {
            return;
        }
        List<LearningSample> historySnapshot;
        synchronized (this) {
            historySnapshot = List.copyOf(history);
        }
        StateFile state = new StateFile(STATE_MAGIC, STATE_VERSION, weights.length,
                generation, weights.clone(), historySnapshot);
        try {
            Path target = stateFile.toAbsolutePath();
            Path parent = target.getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
            Path tempFile = target.resolveSibling(target.getFileName() + ".tmp");
            Files.writeString(tempFile, MAPPER.writeValueAsString(state));
            try {
                Files.move(tempFile, target, StandardCopyOption.ATOMIC_MOVE);
            } catch (AtomicMoveNotSupportedException error) {
                Files.move(tempFile, target, StandardCopyOption.REPLACE_EXISTING);
            }
        } catch (IOException error) {
            logger.warn("learning predictor state save failed: file={}, error={}",
                    stateFile, error.toString());
        }
    }

    private List<LearningSample> retainLatest(List<LearningSample> samples) {
        List<LearningSample> usable = new ArrayList<>();
        if (samples != null) {
            for (LearningSample sample : samples) {
                if (sample != null && sample.features() != null && sample.actualMs() >= 0) {
                    usable.add(sample);
                }
            }
        }
        int fromIndex = Math.max(0, usable.size() - historyLimit);
        return List.copyOf(usable.subList(fromIndex, usable.size()));
    }

    /** Returns a defensive copy of the saved weights, or null when unusable. */
    private static double[] usableWeights(StateFile state) {
        double[] weights = state.weights();
        if (weights == null || weights.length == 0 || state.paramCount() != weights.length) {
            return null;
        }
        for (double weight : weights) {
            if (!Double.isFinite(weight)) {
                return null;
            }
        }
        return weights.clone();
    }
}
