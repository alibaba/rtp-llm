package org.flexlb.config;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.util.List;

/**
 * Converts supported startup documents to the current public schema before
 * Jackson binds the runtime configuration model.
 *
 * <p>The runtime model contains only schema-v2 fields. This class is the single
 * compatibility boundary for schema-v1 documents and deliberately works on the
 * parsed JSON tree so duplicate-key, null, unknown-field, and scalar-type checks
 * remain strict.
 */
final class FlexlbConfigStartupMigrator {

    private static final int SCHEMA_V1 = 1;
    private static final String SCHEMA_VERSION = "schemaVersion";
    private static final String SCHEDULER = "scheduler";
    private static final String DISPATCHER = "dispatcher";
    private static final String DECISION = "decision";
    private static final String CAPACITY = "capacity";
    private static final String TYPE = "type";
    private static final String QUEUE = "QUEUE";
    private static final String DIRECT = "DIRECT";
    private static final String BATCH = "BATCH";
    private static final String NON_BATCH = "NON_BATCH";
    private static final String SINGLE = "SINGLE";
    private static final String FIXED_WINDOW = "FIXED_WINDOW";
    private static final int V1_DEFAULT_MAX_REQUESTS = 8;
    private static final long V1_DEFAULT_MAX_COLLECTION_WAIT_MS = 300L;
    private static final int V1_DEFAULT_MAX_WAITING_REQUESTS_PER_WORKER = 1024;

    private static final List<FieldMove> LEGACY_FIXED_WINDOW_FIELDS = List.of(
            new FieldMove("maxRequests", "maxRequests"),
            new FieldMove("maxCollectionWaitMs", "maxCollectionWaitMs"));
    private static final String LEGACY_PREDICTION_TRIGGER =
            "earlyDispatchPredictedExecutionMs";
    private static final FieldMove LEGACY_PROJECTED_DRAIN_MULTIPLIER =
            new FieldMove(
                    "maxWaitVsAverageMultiplier",
                    "maxProjectedDrainVsAverageMultiplier");
    private static final FieldMove LEGACY_WAITING_CAPACITY = new FieldMove(
            "maxWaitingRequestsPerPrefillWorker",
            "maxWaitingRequestsPerPrefillWorker");

    static MigrationResult migrateToCurrentSchema(JsonNode document) {
        if (!(document instanceof ObjectNode root)) {
            return MigrationResult.unchanged(document);
        }
        JsonNode schemaVersion = root.get(SCHEMA_VERSION);
        if (schemaVersion == null || !schemaVersion.isIntegralNumber()
                || !schemaVersion.canConvertToInt()) {
            // Omission keeps the v2 default; the strict binder rejects malformed types.
            return MigrationResult.unchanged(document);
        }

        int sourceVersion = schemaVersion.intValue();
        if (sourceVersion == FlexlbConfig.CURRENT_SCHEMA_VERSION) {
            return MigrationResult.unchanged(document);
        }
        if (sourceVersion != SCHEMA_V1) {
            throw new ConfigValidationException(SCHEMA_VERSION,
                    "unsupported schema version " + sourceVersion
                            + "; supported versions are " + SCHEMA_V1 + " and "
                            + FlexlbConfig.CURRENT_SCHEMA_VERSION);
        }

        ObjectNode migrated = root.deepCopy();
        migrateSchemaV1(migrated);
        migrated.put(SCHEMA_VERSION, FlexlbConfig.CURRENT_SCHEMA_VERSION);
        return new MigrationResult(migrated, SCHEMA_V1);
    }

    private static void migrateSchemaV1(ObjectNode root) {
        // Routing evolves independently from scheduler/dispatcher shape and
        // must be migrated even when those branches return early.
        migrateV1Routing(root);
        ObjectNode scheduler = optionalObject(root, SCHEDULER);
        ObjectNode dispatcher = optionalObject(root, DISPATCHER);
        String schedulerType = taggedType(scheduler, SCHEDULER, QUEUE);
        String dispatcherType = taggedType(dispatcher, DISPATCHER, BATCH);

        boolean hasLegacyFixedWindowFields = hasAnyLegacyFixedWindowField(dispatcher);
        boolean hasLegacyPredictionTrigger = dispatcher != null
                && dispatcher.has(LEGACY_PREDICTION_TRIGGER);
        boolean hasLegacyWaitingCapacity = dispatcher != null
                && dispatcher.has(LEGACY_WAITING_CAPACITY.source());
        boolean hasAnyLegacyDispatcherField = hasLegacyFixedWindowFields
                || hasLegacyPredictionTrigger
                || hasLegacyWaitingCapacity;

        if (DIRECT.equals(schedulerType)) {
            reject(hasAnyLegacyDispatcherField, DISPATCHER,
                    "schema-v1 dispatcher scheduling fields require scheduler.type=QUEUE");
            return;
        }
        if (!QUEUE.equals(schedulerType)) {
            reject(hasAnyLegacyDispatcherField, SCHEDULER + "." + TYPE,
                    "cannot migrate dispatcher scheduling fields for scheduler type "
                            + display(schedulerType));
            return;
        }
        ObjectNode queueScheduler = ensureQueueScheduler(root, scheduler);
        prepareExplicitV1Decision(queueScheduler);
        if (NON_BATCH.equals(dispatcherType)) {
            reject(hasAnyLegacyDispatcherField, DISPATCHER,
                    "schema-v1 BATCH scheduling fields are not valid for NON_BATCH");
            if (!queueScheduler.has(DECISION)) {
                queueScheduler.set(DECISION, taggedObject(SINGLE));
            }
            materializeV1QueueCapacity(queueScheduler);
            return;
        }
        if (!BATCH.equals(dispatcherType)) {
            reject(hasAnyLegacyDispatcherField, DISPATCHER + "." + TYPE,
                    "cannot migrate scheduling fields for dispatcher type "
                            + display(dispatcherType));
            return;
        }

        validateLegacyBatchFields(dispatcher);

        boolean hasExplicitDecision = queueScheduler.has(DECISION);
        ObjectNode decision = optionalObject(queueScheduler, DECISION);
        if (!hasExplicitDecision) {
            if (hasLegacyPredictionTrigger) {
                throw lossyPredictionBoundary(
                        DISPATCHER + "." + LEGACY_PREDICTION_TRIGGER);
            }
            decision = taggedObject(FIXED_WINDOW);
            decision.put("maxRequests", V1_DEFAULT_MAX_REQUESTS);
            decision.put("maxCollectionWaitMs",
                    V1_DEFAULT_MAX_COLLECTION_WAIT_MS);
            queueScheduler.set(DECISION, decision);
            for (FieldMove move : LEGACY_FIXED_WINDOW_FIELDS) {
                replaceField(dispatcher, decision, move);
            }
        } else {
            // In schema v1 an explicit decision owned these values. The legacy
            // dispatcher fields were validated but did not affect behavior.
            removeFields(dispatcher, LEGACY_FIXED_WINDOW_FIELDS);
            if (dispatcher != null) {
                dispatcher.remove(LEGACY_PREDICTION_TRIGGER);
            }
        }
        if (hasLegacyWaitingCapacity) {
            ObjectNode capacity = ensureObject(queueScheduler, CAPACITY,
                    SCHEDULER + "." + CAPACITY);
            moveFieldUnlessTargetPresent(
                    dispatcher, capacity, LEGACY_WAITING_CAPACITY);
        }
        materializeV1QueueCapacity(queueScheduler);
    }

    private static void migrateV1Routing(ObjectNode root) {
        String path = "router";
        ObjectNode router = optionalObject(root, "router", path);
        if (router == null) {
            return;
        }
        path += ".roles";
        ObjectNode roles = optionalObject(router, "roles", path);
        if (roles == null) {
            return;
        }
        path += ".prefill";
        ObjectNode prefill = optionalObject(roles, "prefill", path);
        if (prefill == null) {
            return;
        }
        path += ".selector";
        ObjectNode selector = optionalObject(prefill, "selector", path);
        if (selector == null) {
            return;
        }
        path += ".candidateChoice";
        ObjectNode candidateChoice = optionalObject(
                selector, "candidateChoice", path);
        if (candidateChoice == null) {
            return;
        }
        path += ".outlierRejection";
        ObjectNode outlier = optionalObject(
                candidateChoice, "outlierRejection", path);
        if (outlier != null) {
            moveFieldUnlessTargetPresent(
                    outlier, outlier,
                    LEGACY_PROJECTED_DRAIN_MULTIPLIER);
        }
    }

    private static ObjectNode ensureQueueScheduler(ObjectNode root,
                                                   ObjectNode scheduler) {
        if (scheduler != null) {
            return scheduler;
        }
        ObjectNode created = taggedObject(QUEUE);
        root.set(SCHEDULER, created);
        return created;
    }

    private static ObjectNode ensureObject(ObjectNode parent,
                                           String field,
                                           String path) {
        JsonNode existing = parent.get(field);
        if (existing == null) {
            ObjectNode created = JsonNodeFactory.instance.objectNode();
            parent.set(field, created);
            return created;
        }
        if (existing instanceof ObjectNode object) {
            return object;
        }
        throw new ConfigValidationException(path, "must be a JSON object");
    }

    private static ObjectNode optionalObject(ObjectNode parent, String field) {
        return optionalObject(parent, field, field);
    }

    private static ObjectNode optionalObject(ObjectNode parent,
                                             String field,
                                             String path) {
        JsonNode value = parent.get(field);
        if (value == null) {
            return null;
        }
        if (value instanceof ObjectNode object) {
            return object;
        }
        throw new ConfigValidationException(path, "must be a JSON object");
    }

    private static String taggedType(ObjectNode object,
                                     String path,
                                     String defaultType) {
        if (object == null) {
            return defaultType;
        }
        JsonNode type = object.get(TYPE);
        if (type == null) {
            return null;
        }
        if (!type.isTextual()) {
            throw new ConfigValidationException(path + "." + TYPE,
                    "must be a string");
        }
        return type.textValue();
    }

    private static boolean hasAnyLegacyFixedWindowField(ObjectNode dispatcher) {
        if (dispatcher == null) {
            return false;
        }
        for (FieldMove field : LEGACY_FIXED_WINDOW_FIELDS) {
            if (dispatcher.has(field.source())) {
                return true;
            }
        }
        return false;
    }

    private static void moveFieldUnlessTargetPresent(ObjectNode source,
                                                     ObjectNode target,
                                                     FieldMove field) {
        if (!source.has(field.source())) {
            return;
        }
        if (target.has(field.target())) {
            source.remove(field.source());
            return;
        }
        target.set(field.target(), source.remove(field.source()));
    }

    private static void replaceField(ObjectNode source,
                                     ObjectNode target,
                                     FieldMove field) {
        if (source != null && source.has(field.source())) {
            target.set(field.target(), source.remove(field.source()));
        }
    }

    private static void materializeV1QueueCapacity(ObjectNode queueScheduler) {
        ObjectNode capacity = ensureObject(queueScheduler, CAPACITY,
                SCHEDULER + "." + CAPACITY);
        if (!capacity.has(LEGACY_WAITING_CAPACITY.target())) {
            capacity.put(LEGACY_WAITING_CAPACITY.target(),
                    V1_DEFAULT_MAX_WAITING_REQUESTS_PER_WORKER);
        }
    }

    private static void materializeV1FixedWindowDefaults(ObjectNode decision) {
        if (!decision.has("maxRequests")) {
            decision.put("maxRequests", V1_DEFAULT_MAX_REQUESTS);
        }
        if (!decision.has("maxCollectionWaitMs")) {
            decision.put("maxCollectionWaitMs",
                    V1_DEFAULT_MAX_COLLECTION_WAIT_MS);
        }
    }

    private static void prepareExplicitV1Decision(ObjectNode queueScheduler) {
        ObjectNode decision = optionalObject(queueScheduler, DECISION);
        if (decision == null) {
            return;
        }
        String decisionType = taggedType(
                decision, SCHEDULER + "." + DECISION, null);
        if (FIXED_WINDOW.equals(decisionType)
                && decision.has("maxPredictedExecutionMs")) {
            throw lossyPredictionBoundary(
                    SCHEDULER + "." + DECISION
                            + ".maxPredictedExecutionMs");
        }
        if (FIXED_WINDOW.equals(decisionType)) {
            materializeV1FixedWindowDefaults(decision);
        }
    }

    private static void removeFields(ObjectNode object, List<FieldMove> fields) {
        if (object == null) {
            return;
        }
        for (FieldMove field : fields) {
            object.remove(field.source());
        }
    }

    private static void validateLegacyBatchFields(ObjectNode dispatcher) {
        if (dispatcher == null) {
            return;
        }
        validatePositiveInt(dispatcher, "maxRequests");
        validateNonNegativeLong(dispatcher, "maxCollectionWaitMs");
        validatePositiveInt(dispatcher,
                LEGACY_WAITING_CAPACITY.source());
        validatePositiveLong(dispatcher, LEGACY_PREDICTION_TRIGGER);
    }

    private static void validatePositiveInt(ObjectNode object, String field) {
        JsonNode value = object.get(field);
        if (value == null) {
            return;
        }
        if (!value.isIntegralNumber() || !value.canConvertToInt()) {
            throw new ConfigValidationException(DISPATCHER + "." + field,
                    "must be a 32-bit integer");
        }
        if (value.intValue() <= 0) {
            throw new ConfigValidationException(DISPATCHER + "." + field,
                    "must be greater than zero");
        }
    }

    private static void validatePositiveLong(ObjectNode object, String field) {
        JsonNode value = object.get(field);
        if (value == null) {
            return;
        }
        if (!value.isIntegralNumber() || !value.canConvertToLong()) {
            throw new ConfigValidationException(DISPATCHER + "." + field,
                    "must be a 64-bit integer");
        }
        if (value.longValue() <= 0L) {
            throw new ConfigValidationException(DISPATCHER + "." + field,
                    "must be greater than zero");
        }
    }

    private static void validateNonNegativeLong(ObjectNode object, String field) {
        JsonNode value = object.get(field);
        if (value == null) {
            return;
        }
        if (!value.isIntegralNumber() || !value.canConvertToLong()) {
            throw new ConfigValidationException(DISPATCHER + "." + field,
                    "must be a 64-bit integer");
        }
        if (value.longValue() < 0L) {
            throw new ConfigValidationException(DISPATCHER + "." + field,
                    "must be non-negative");
        }
    }

    private static ConfigValidationException lossyPredictionBoundary(String field) {
        return new ConfigValidationException(field,
                "cannot be migrated without changing its equality-boundary "
                        + "dispatch behavior; use schemaVersion=2 and confirm "
                        + "the inclusive prediction cap explicitly");
    }

    private static ObjectNode taggedObject(String type) {
        ObjectNode object = JsonNodeFactory.instance.objectNode();
        object.put(TYPE, type);
        return object;
    }

    private static String display(String value) {
        return value == null ? "<missing>" : value;
    }

    private static void reject(boolean condition, String field, String message) {
        if (condition) {
            throw new ConfigValidationException(field, message);
        }
    }

    private record FieldMove(String source, String target) {
    }

    record MigrationResult(JsonNode document, Integer sourceSchemaVersion) {

        static MigrationResult unchanged(JsonNode document) {
            return new MigrationResult(document, null);
        }

        boolean migrated() {
            return sourceSchemaVersion != null;
        }
    }

    private FlexlbConfigStartupMigrator() {
    }
}
