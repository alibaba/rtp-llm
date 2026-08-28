package org.flexlb.config;

import org.flexlb.config.RoutingConfig.FormulaEstimatorConfig;
import org.flexlb.config.RoutingConfig.LearningEstimatorConfig;
import org.flexlb.config.RoutingConfig.LearningPersistenceConfig;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Configuration-surface tests for the LEARNING estimator persistence block:
 * defaults, strict-schema binding, validator rules and FORMULA compatibility.
 * The PrefillEndpoint wiring itself is not unit-constructible, so coverage
 * stops at the parsed and validated config layer.
 */
class PersistenceConfigTest {

    @Test
    @DisplayName("LEARNING 估算器省略 persistence 时使用禁用默认值")
    void omittedPersistenceUsesDisabledDefaults() {
        FlexlbConfig config = ConfigService.parse("""
                {"router":{"roles":{"prefill":{"executionTimeEstimator":
                  {"type":"LEARNING"}}}}}
                """);
        LearningEstimatorConfig estimator = assertInstanceOf(LearningEstimatorConfig.class,
                config.getRouter().getRoles().getPrefill().getExecutionTimeEstimator());
        LearningPersistenceConfig persistence = estimator.getPersistence();
        assertFalse(persistence.isEnabled(), "persistence must default to disabled");
        assertEquals(2000, persistence.getHistoryLimit(),
                "historyLimit must default to the documented rolling window");
        assertEquals(10, persistence.getRefitEpochs(),
                "refitEpochs must default to the tuned cold-start refit schedule");
        assertEquals(256, persistence.getSaveInterval(),
                "saveInterval must default to the documented throttle window");
        assertNull(persistence.getStateFile(),
                "stateFile must default to null so each endpoint derives its own path");

        FlexlbConfig defaults = new ConfigService(Map.of()).loadBalanceConfig();
        assertInstanceOf(FormulaEstimatorConfig.class,
                defaults.getRouter().getRoles().getPrefill().getExecutionTimeEstimator(),
                "the default estimator must stay FORMULA when FLEXLB_CONFIG is absent");
    }

    @Test
    @DisplayName("合法 LEARNING persistence 配置经严格 schema 全链路绑定")
    void learningPersistenceDocumentBindsThroughStrictSchema() {
        FlexlbConfig config = ConfigService.parse("""
                {"router":{"roles":{"prefill":{"executionTimeEstimator":{
                  "type":"LEARNING",
                  "persistence":{
                    "enabled":true,
                    "stateFile":"/tmp/flexlb/learning-predictor/state.json",
                    "historyLimit":512,
                    "refitEpochs":40,
                    "saveInterval":64}}}}}}
                """);
        LearningEstimatorConfig estimator = assertInstanceOf(LearningEstimatorConfig.class,
                config.getRouter().getRoles().getPrefill().getExecutionTimeEstimator());
        LearningPersistenceConfig persistence = estimator.getPersistence();
        assertTrue(persistence.isEnabled(), "enabled must bind from the document");
        assertEquals("/tmp/flexlb/learning-predictor/state.json", persistence.getStateFile(),
                "stateFile must bind from the document");
        assertEquals(512, persistence.getHistoryLimit(),
                "historyLimit must bind from the document");
        assertEquals(40, persistence.getRefitEpochs(),
                "refitEpochs must bind from the document");
        assertEquals(64, persistence.getSaveInterval(),
                "saveInterval must bind from the document");

        ConfigValidationException unknownField = assertThrows(ConfigValidationException.class,
                () -> ConfigService.parse("""
                        {"router":{"roles":{"prefill":{"executionTimeEstimator":{
                          "type":"LEARNING",
                          "persistence":{"enabled":true,"foo":1}}}}}}
                        """));
        assertTrue(unknownField.getMessage().contains("Invalid FLEXLB_CONFIG JSON"),
                "unknown persistence fields must be rejected by the strict schema: "
                        + unknownField.getMessage());
    }

    @Test
    @DisplayName("Validator 拒绝非法 persistence 取值且 refitEpochs=0 合法")
    void validatorRejectsIllegalPersistenceValues() {
        assertRejected("{\"historyLimit\":0}", "persistence.historyLimit");
        assertRejected("{\"historyLimit\":-5}", "persistence.historyLimit");
        assertRejected("{\"refitEpochs\":-1}", "persistence.refitEpochs");
        assertRejected("{\"saveInterval\":0}", "persistence.saveInterval");
        assertRejected("{\"stateFile\":\"\"}", "persistence.stateFile");
        assertRejected("{\"stateFile\":\"   \"}", "persistence.stateFile");

        FlexlbConfig zeroRefit = ConfigService.parse("""
                {"router":{"roles":{"prefill":{"executionTimeEstimator":{
                  "type":"LEARNING",
                  "persistence":{"refitEpochs":0}}}}}}
                """);
        LearningEstimatorConfig estimator = assertInstanceOf(LearningEstimatorConfig.class,
                zeroRefit.getRouter().getRoles().getPrefill().getExecutionTimeEstimator());
        assertEquals(0, estimator.getPersistence().getRefitEpochs(),
                "refitEpochs=0 is non-negative and must stay legal");

        ConfigValidationException nullPersistence = assertThrows(
                ConfigValidationException.class, () -> ConfigService.parse("""
                        {"router":{"roles":{"prefill":{"executionTimeEstimator":{
                          "type":"LEARNING","persistence":null}}}}}
                        """));
        assertTrue(nullPersistence.getMessage().contains("null is not allowed"),
                "a JSON null persistence block must be rejected outright: "
                        + nullPersistence.getMessage());
    }

    @Test
    @DisplayName("FORMULA 估算器不带 persistence 字段保持兼容且互斥字段被拒")
    void formulaEstimatorWithoutPersistenceStaysCompatible() {
        FlexlbConfig config = ConfigService.parse("""
                {"router":{"roles":{"prefill":{"executionTimeEstimator":{
                  "type":"FORMULA","expression":"sum(computeTokens)"}}}}}
                """);
        FormulaEstimatorConfig estimator = assertInstanceOf(FormulaEstimatorConfig.class,
                config.getRouter().getRoles().getPrefill().getExecutionTimeEstimator());
        assertEquals("sum(computeTokens)", estimator.getExpression(),
                "the FORMULA variant must keep parsing exactly as before");

        assertThrows(ConfigValidationException.class, () -> ConfigService.parse("""
                {"router":{"roles":{"prefill":{"executionTimeEstimator":{
                  "type":"FORMULA",
                  "expression":"sum(computeTokens)",
                  "persistence":{"enabled":true}}}}}}
                """), "persistence belongs to the LEARNING variant only and must be"
                + " rejected as an unknown field on FORMULA");
    }

    private static void assertRejected(String persistenceJson, String field) {
        String document = "{\"router\":{\"roles\":{\"prefill\":{\"executionTimeEstimator\":"
                + "{\"type\":\"LEARNING\",\"persistence\":" + persistenceJson + "}}}}}";
        ConfigValidationException error = assertThrows(ConfigValidationException.class,
                () -> ConfigService.parse(document),
                "illegal persistence values must fail validation: " + persistenceJson);
        assertTrue(error.getMessage().contains(field),
                "the error must name the offending field '" + field + "': "
                        + error.getMessage());
    }
}
