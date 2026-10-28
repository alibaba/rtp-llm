package org.flexlb.mock.compute;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.databind.*;
import com.fasterxml.jackson.databind.json.JsonMapper;
import java.io.IOException;
import java.util.*;
import org.flexlb.mock.api.model.ComputeSpec;
import org.flexlb.mock.api.model.Types.Role;

/** One component owns one formula and noise model; there is no role map or fallback calibration. */
public final class ComputeModel {
    private static final ObjectMapper JSON = JsonMapper.builder()
        .enable(JsonParser.Feature.STRICT_DUPLICATE_DETECTION)
        .enable(DeserializationFeature.FAIL_ON_TRAILING_TOKENS)
        .disable(DeserializationFeature.ACCEPT_FLOAT_AS_INT)
        .disable(MapperFeature.ALLOW_COERCION_OF_SCALARS)
        .serializationInclusion(JsonInclude.Include.NON_NULL).build();
    private final ComputeSpec configuration;
    private final ForwardModel forward;
    private final NoiseModel noise;

    public ComputeModel(ComputeSpec spec) {
        configuration = Objects.requireNonNull(spec);
        forward = new FormulaForward(spec.expression()); noise = new NoiseModel(spec.noise());
    }

    public static ComputeModel fromJson(JsonNode node) throws IOException {
        rejectNull(node);
        return new ComputeModel(JSON.treeToValue(node, ComputeSpec.class));
    }

    public ComputeSpec configuration() { return configuration; }
    public JsonNode effectiveConfiguration() { return JSON.valueToTree(configuration); }

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
