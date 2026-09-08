package org.flexlb.config;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/** Explicit offline migration. The online configuration loader accepts only v2. */
public final class FlexlbConfigMigration {
    private static final ObjectMapper JSON = new ObjectMapper();

    private FlexlbConfigMigration() { }

    public record Result(String document, List<String> behaviorChanges) {
        public Result {
            behaviorChanges = List.copyOf(behaviorChanges);
        }
    }

    public static Result fromV1(String document) throws IOException {
        JsonNode parsed = JSON.readTree(document);
        if (!(parsed instanceof ObjectNode root)
                || integer(root, "schemaVersion", "schemaVersion", 1L) != 1L) {
            throw new ConfigValidationException("schemaVersion", "offline migration requires v1");
        }
        List<String> changes = new ArrayList<>();
        ObjectNode scheduler = object(root, "scheduler");
        String schedulerType = scheduler.path("type").asText("QUEUE");
        ObjectNode dispatcher = object(root, "dispatcher");
        String delivery = dispatcher.path("type").asText("BATCH");
        if (schedulerType.equals("QUEUE")) {
            if (scheduler.has("decision")) {
                throw new ConfigValidationException("scheduler.decision", "is not a v1 field");
            }
            ObjectNode decision = object(scheduler, "decision");
            decision.put("type", delivery.equals("NON_BATCH") ? "SINGLE" : "FIXED_WINDOW");
            if (delivery.equals("BATCH")) {
                move(dispatcher, "maxRequests", decision, "maxRequests");
                move(dispatcher, "maxCollectionWaitMs", decision, "maxCollectionWaitMs");
                move(dispatcher, "earlyDispatchPredictedExecutionMs", decision, "maxPredictedExecutionMs");
                move(dispatcher, "maxWaitingRequestsPerPrefillWorker",
                        object(scheduler, "capacity"), "maxWaitingRequestsPerPrefillWorker");
            }
        }

        ObjectNode router = object(root, "router");
        long hysteresis = integer(router, "availabilityHysteresisPercent",
                "router.availabilityHysteresisPercent", 15L);
        if (hysteresis < 0L || hysteresis > 100L) {
            throw new ConfigValidationException("router.availabilityHysteresisPercent", "must be in [0, 100]");
        }
        router.remove("availabilityHysteresisPercent");
        if (hysteresis != 0L) {
            changes.add("Availability hysteresis " + hysteresis
                    + "% has no v2 equivalent; admission recovers when exact capacity is released.");
        }
        ObjectNode roles = object(router, "roles");
        ObjectNode prefill = object(roles, "prefill");
        JsonNode availability = prefill.remove("availability");
        if (availability != null) {
            requireOnly(availability, "router.roles.prefill.availability", "maxPendingRequests");
        }
        long pendingLimit = availability == null ? 64L : integer(availability, "maxPendingRequests",
                "router.roles.prefill.availability.maxPendingRequests", 64L);
        if (pendingLimit <= 0 || pendingLimit > Integer.MAX_VALUE) {
            throw new ConfigValidationException("router.roles.prefill.availability.maxPendingRequests",
                    "must be a positive integer");
        }
        if (delivery.equals("NON_BATCH")) {
            long configured = integer(dispatcher, "maxInflightRequestsPerPrefillWorker",
                    "dispatcher.maxInflightRequestsPerPrefillWorker", pendingLimit);
            if (configured <= 0 || configured > Integer.MAX_VALUE) {
                throw new ConfigValidationException("dispatcher.maxInflightRequestsPerPrefillWorker",
                        "must be a positive integer");
            }
            dispatcher.put("maxInflightRequestsPerPrefillWorker", Math.min(configured, pendingLimit));
            changes.add("NON_BATCH capacity counts canonical outstanding requests plus unowned Engine work;"
                    + " the old pending limit is mapped to maxInflightRequestsPerPrefillWorker.");
        } else {
            changes.add("The old Prefill pending limit " + pendingLimit
                    + " is not equivalent to a BATCH queue or batch-inflight limit; size those capacities explicitly.");
        }
        JsonNode prefillSelector = prefill.remove("selector");
        if (prefillSelector != null) {
            requireOnly(prefillSelector, "router.roles.prefill.selector", "type", "candidateChoice");
            requireSelector(prefillSelector, "ESTIMATED_TTFT", "router.roles.prefill.selector.type");
            if (prefillSelector.has("candidateChoice")) {
                move((ObjectNode) prefillSelector, "candidateChoice", prefill, "candidateChoice");
            }
        }
        JsonNode outliers = prefill.path("candidateChoice").path("outlierRejection");
        if (outliers instanceof ObjectNode object) {
            move(object, "maxWaitVsAverageMultiplier", object, "maxProjectedDrainVsAverageMultiplier");
        }
        ObjectNode decode = object(roles, "decode");
        JsonNode decodeSelector = decode.remove("selector");
        if (decodeSelector != null) {
            requireOnly(decodeSelector, "router.roles.decode.selector", "type", "decayPerToken", "outlierRejection");
            requireSelector(decodeSelector, "KV_USAGE_WEIGHTED_RANDOM", "router.roles.decode.selector.type");
            move((ObjectNode) decodeSelector, "decayPerToken", decode, "decayPerToken");
            move((ObjectNode) decodeSelector, "outlierRejection", decode, "outlierRejection");
        }
        JsonNode vit = roles.remove("vit");
        if (vit != null) {
            requireOnly(vit, "router.roles.vit", "selector");
            if (vit.has("selector")) {
                requireOnly(vit.get("selector"), "router.roles.vit.selector", "type");
                requireSelector(vit.get("selector"), "RANDOM", "router.roles.vit.selector.type");
            }
        }
        root.put("schemaVersion", 2);
        String migrated = JSON.writerWithDefaultPrettyPrinter().writeValueAsString(root);
        ConfigService.parse(migrated);
        changes.add("QUEUE Decode acceptance capacity is shared across FIFO/PRIORITY and charged at delivery preparation.");
        return new Result(migrated, changes);
    }

    private static ObjectNode object(ObjectNode parent, String field) {
        JsonNode existing = parent.get(field);
        if (existing == null) {
            return parent.putObject(field);
        }
        if (existing instanceof ObjectNode object) {
            return object;
        }
        throw new ConfigValidationException(field, "must be an object");
    }

    private static long integer(JsonNode object, String field, String path, long defaultValue) {
        JsonNode value = object.get(field);
        if (value == null) {
            return defaultValue;
        }
        if (!value.isIntegralNumber() || !value.canConvertToLong()) {
            throw new ConfigValidationException(path, "must be an integer");
        }
        return value.longValue();
    }

    private static void move(ObjectNode source, String from, ObjectNode target, String to) {
        JsonNode value = source.remove(from);
        if (value != null) {
            if (target.has(to)) {
                throw new ConfigValidationException(to, "conflicts with migrated field " + from);
            }
            target.set(to, value);
        }
    }

    private static void requireOnly(JsonNode object, String path, String... allowed) {
        if (!object.isObject()) {
            throw new ConfigValidationException(path, "must be an object");
        }
        var names = java.util.Set.of(allowed);
        object.fieldNames().forEachRemaining(field -> {
            if (!names.contains(field)) {
                throw new ConfigValidationException(path + "." + field, "cannot be migrated");
            }
        });
    }

    private static void requireSelector(JsonNode selector, String supported, String path) {
        if (!selector.path("type").asText().equals(supported)) {
            throw new ConfigValidationException(path,
                    "has no equivalent v2 policy: " + selector.path("type").asText());
        }
    }

    /** Read v1 from stdin; write v2 to stdout and behavioral differences to stderr. */
    public static void main(String[] args) throws IOException {
        Result result = fromV1(new String(System.in.readAllBytes(), StandardCharsets.UTF_8));
        result.behaviorChanges().forEach(System.err::println);
        System.out.println(result.document());
    }
}
