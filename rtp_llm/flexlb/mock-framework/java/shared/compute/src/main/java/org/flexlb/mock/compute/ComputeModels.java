package org.flexlb.mock.compute;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.databind.*;
import com.fasterxml.jackson.databind.json.JsonMapper;
import java.io.IOException;
import java.nio.file.Path;
import java.util.*;
import org.flexlb.mock.api.model.ComputeSpec;
import org.flexlb.mock.api.model.Types.Role;

/** Immutable role models assembled from environment.compute. Missing models never invent a calibration. */
public final class ComputeModels {
    private record Model(ForwardModel forward, NoiseModel noise) {}
    private static final ObjectMapper JSON = JsonMapper.builder()
        .enable(JsonParser.Feature.STRICT_DUPLICATE_DETECTION)
        .enable(DeserializationFeature.FAIL_ON_TRAILING_TOKENS)
        .disable(DeserializationFeature.ACCEPT_FLOAT_AS_INT)
        .disable(MapperFeature.ALLOW_COERCION_OF_SCALARS)
        .serializationInclusion(JsonInclude.Include.NON_NULL).build();
    private final ComputeSpec configuration;
    private final Map<Role, Model> models;

    public ComputeModels(ComputeSpec spec) {
        configuration = Objects.requireNonNull(spec);
        var values = new EnumMap<Role, Model>(Role.class);
        for (var role : Role.values()) {
            var model = switch (role) { case prefill -> spec.prefill(); case decode -> spec.decode(); case vit -> spec.vit(); };
            if (model != null) values.put(role, new Model(new FormulaForward(model.expression()), new NoiseModel(model.noise())));
        }
        models = Map.copyOf(values);
    }

    public static ComputeModels fromJson(JsonNode node) throws IOException {
        rejectNull(node);
        return new ComputeModels(JSON.treeToValue(node, ComputeSpec.class));
    }

    /** Reads the runner's component config file, with the original native business values left intact. */
    public static ComputeModels fromComponentFile(Path path) throws IOException {
        JsonNode root = JSON.readTree(path.toFile());
        var node = root.path("environment").path("compute");
        if (node.isMissingNode()) throw new IllegalArgumentException("Missing environment.compute");
        return fromJson(node);
    }

    public ComputeSpec configuration() { return configuration; }
    public JsonNode effectiveConfiguration() { return JSON.valueToTree(configuration); }

    public NoiseModel.Sample plan(Role role, String engineId, ForwardBatch batch) {
        var model = models.get(role);
        if (model == null) throw new IllegalArgumentException("No compute model for " + role);
        if (engineId == null || engineId.isBlank()) throw new IllegalArgumentException("Engine id is required");
        String key = engineId.length() + ":" + engineId + "/" + role + "/" + batch.id();
        return model.noise.apply(model.forward.estimate(batch), key);
    }

    private static void rejectNull(JsonNode node) {
        if (node == null || node.isNull() || node.isMissingNode()) throw new IllegalArgumentException("Compute configuration cannot be null");
        node.forEach(ComputeModels::rejectNull);
    }
}
