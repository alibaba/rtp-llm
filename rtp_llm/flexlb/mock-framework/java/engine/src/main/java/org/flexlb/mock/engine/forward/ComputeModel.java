package org.flexlb.mock.engine.forward;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.databind.*;
import com.fasterxml.jackson.databind.json.JsonMapper;
import java.io.IOException;
import java.util.*;
import org.flexlb.mock.api.model.ComputeSpec;
import org.flexlb.mock.api.model.Types.Role;

/** Composes Forward timing, Decode acceptance and EOS/length completion without mutating request state. */
public final class ComputeModel {
    private static final ObjectMapper JSON = JsonMapper.builder()
        .enable(JsonParser.Feature.STRICT_DUPLICATE_DETECTION)
        .enable(DeserializationFeature.FAIL_ON_TRAILING_TOKENS)
        .disable(DeserializationFeature.ACCEPT_FLOAT_AS_INT)
        .disable(MapperFeature.ALLOW_COERCION_OF_SCALARS)
        .serializationInclusion(JsonInclude.Include.NON_NULL).build();
    private final ComputeSpec configuration;
    private final FormulaForward forward;
    private final NoiseModel noise;
    private final DecodeStepModel decode;
    private final StoppingModel stopping;

    public record Request(String key, long round, int inputTokens, int generated, int minimum,
                          int maximum, int maxSequence, boolean ignoreEos) {
        public Request {
            if (key == null || key.isBlank() || round < 0 || inputTokens < 1 || generated < 0 || minimum < 0
                    || maximum < 1 || minimum > maximum || maxSequence <= inputTokens
                    || generated >= Math.min(maximum, maxSequence - inputTokens))
                throw new IllegalArgumentException("Invalid generation request/limits");
        }
        int limit() { return Math.min(maximum, maxSequence - inputTokens); }
    }
    public record Plan(Request request, DecodeStepModel.Plan progress) {
        public int verifyTokens() { return progress.verifyTokens(); }
        public int additionalKvTokens() { return progress.additionalKvTokens(); }
    }
    public record Result(int committedTokens, boolean finished, int candidates) {
        public Result { if (committedTokens < 1 || candidates < committedTokens) throw new IllegalArgumentException("Invalid generation result"); }
    }

    public ComputeModel(ComputeSpec spec) {
        this(spec, Map.of());
    }

    public ComputeModel(ComputeSpec spec, Map<String, Double> constants) {
        configuration = Objects.requireNonNull(spec);
        forward = new FormulaForward(spec.forward().expression(), constants); noise = new NoiseModel(spec.forward().noise());
        decode = new DecodeStepModel(spec.decode()); stopping = new StoppingModel(spec.eos());
    }

    public static ComputeModel fromJson(JsonNode node) throws IOException {
        return fromJson(node, Map.of());
    }

    public static ComputeModel fromJson(JsonNode node, Map<String, Double> constants) throws IOException {
        rejectNull(node);
        return new ComputeModel(JSON.treeToValue(node, ComputeSpec.class), constants);
    }

    private static final Map<String, String> SETTINGS = Map.ofEntries(
        Map.entry("forward.expression", "forward.expression"),
        Map.entry("forward.noise.stddev.ms", "forward.noise.stddev_ms"),
        Map.entry("forward.noise.stddev.ratio", "forward.noise.stddev_ratio"),
        Map.entry("forward.noise.max.deviation.ms", "forward.noise.max_deviation_ms"),
        Map.entry("forward.noise.seed", "forward.noise.seed"),
        Map.entry("decode.draft.tokens", "decode.draft_tokens"),
        Map.entry("decode.acceptance.rate", "decode.acceptance_rate"),
        Map.entry("decode.tokens.per.step", "decode.tokens_per_step"),
        Map.entry("decode.seed", "decode.seed"),
        Map.entry("eos.probability", "eos.probability"),
        Map.entry("eos.seed", "eos.seed"));

    public static Set<String> settingNames() {
        var names = new HashSet<String>();
        SETTINGS.keySet().forEach(key -> names.add("mock.compute." + key));
        return Set.copyOf(names);
    }

    /** Input is normalized by the component; ENV strings share the record validators. */
    public static ComputeModel fromSettings(JsonNode settings, Map<String, Double> constants) throws IOException {
        var spec = JSON.createObjectNode();
        for (var entry : SETTINGS.entrySet()) {
            var value = settings.get("mock.compute." + entry.getKey());
            if (value == null) continue;
            if (value.isTextual() && !entry.getKey().equals("forward.expression")) {
                value = JSON.readTree(value.textValue());
                if (value == null || !value.isNumber()) throw new IllegalArgumentException("Expected number: mock.compute." + entry.getKey());
            }
            String[] path = entry.getValue().split("\\.");
            var target = spec;
            for (int index = 0; index < path.length - 1; index++) target = target.withObject("/" + path[index]);
            target.set(path[path.length - 1], value);
        }
        return fromJson(spec, constants);
    }

    public JsonNode effectiveSettings() {
        var result = JSON.createObjectNode();
        var spec = effectiveConfiguration();
        SETTINGS.forEach((key, path) -> result.set("mock.compute." + key, spec.at("/" + path.replace('.', '/'))));
        return result;
    }

    public int kvReserveTokens() { return decode.reserveTokens(); }

    public ComputeSpec configuration() { return configuration; }
    public JsonNode effectiveConfiguration() { return JSON.valueToTree(configuration); }

    public Plan prepare(Role role, Request request) {
        if (role != Role.prefill && role != Role.decode) throw new IllegalArgumentException("Unsupported generation role");
        if (role == Role.prefill && request.generated() != 0) throw new IllegalArgumentException("Prefill must produce the first token");
        return new Plan(request, role == Role.prefill ? new DecodeStepModel.Plan(0, 1, 0, 1)
            : decode.plan(request.round(), request.limit() - request.generated()));
    }

    /** Pure result calculation: no request mutation, queue operations or resource release. */
    public Result completed(Plan plan) {
        var request = plan.request();
        int candidates = decode.candidates(plan.progress(), request.key(), request.round());
        return stopping.complete(request.key(), request.generated(), candidates, request.minimum(), request.limit(), request.ignoreEos());
    }

    public NoiseModel.Sample plan(Role role, String engineId, ForwardBatch batch) {
        Objects.requireNonNull(role);
        if (engineId == null || engineId.isBlank()) throw new IllegalArgumentException("Engine id is required");
        String key = engineId.length() + ":" + engineId + "/" + role + "/" + batch.id();
        return noise.apply(forward.estimate(batch), key);
    }

    private static void rejectNull(JsonNode node) {
        if (node == null || node.isNull() || node.isMissingNode()) throw new IllegalArgumentException("Compute configuration cannot be null");
        node.forEach(ComputeModel::rejectNull);
    }
}
