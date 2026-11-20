package org.flexlb.config.loading.parser;

import org.flexlb.config.loading.ConfigSchemaVersion;
import org.flexlb.config.loading.NormalizedConfig;
import org.flexlb.config.loading.source.ConfigSource;
import org.junit.jupiter.api.Test;
import uk.org.webcompere.systemstubs.environment.EnvironmentVariables;

import java.util.function.Consumer;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * Checks legacy configuration parsing and compatibility errors.
 */
class V0ConfigDocumentParserTest {

    @Test
    void resolvesExactlyOneParserFromDocumentVersionBeforeEnvironmentVersion() throws Exception {
        new StandardConfigDocumentParser();
        new V0ConfigDocumentParser();
        assertThat(
                        ConfigDocumentParserResolver.resolve(
                                        "{\"schemaVersion\":0,\"enableQueueing\":true}")
                                .schemaVersion())
                .isEqualTo(ConfigSchemaVersion.V0_COMPATIBILITY);
        new EnvironmentVariables(ConfigDocumentParserResolver.CONFIG_SCHEMA_VERSION_ENV, "0")
                .execute(
                        () -> {
                            assertThat(
                                            ConfigDocumentParserResolver.resolve(
                                                    "{\"schemaVersion\":"
                                                            + ConfigSchemaVersion.STANDARD
                                                            + "}"))
                                    .extracting(ConfigDocumentParser::schemaVersion)
                                    .isEqualTo(ConfigSchemaVersion.STANDARD);
                            assertThat(
                                            ConfigDocumentParserResolver.resolve(
                                                            "{\"enableQueueing\":true}")
                                                    .schemaVersion())
                                    .isEqualTo(ConfigSchemaVersion.V0_COMPATIBILITY);
                        });
    }

    @Test
    void rejectsMalformedDocumentBeforeSelectingFallbackParser() {
        assertThatThrownBy(() -> ConfigDocumentParserResolver.resolve("{not json"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessage("Invalid configuration JSON");
    }

    @Test
    void rejectsMissingV0CompatibilityDocument() {
        V0ConfigDocumentParser parser = new V0ConfigDocumentParser();
        assertThatThrownBy(() -> parser.parse(null, null))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessage("V0 compatibility configuration document must not be null or blank");
        assertThatThrownBy(() -> parser.parse("   ", null))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessage("V0 compatibility configuration document must not be null or blank");
    }

    /**
     * Supplies legacy documents to exercise parser compatibility behavior.
     */
    private static final class CompatibilitySource implements ConfigSource {
        private final NormalizedConfig converted;

        private CompatibilitySource(NormalizedConfig converted) {
            this.converted = converted;
        }

        @Override
        public String name() {
            return "V0 compatibility";
        }

        @Override
        public int priority() {
            return 2;
        }

        @Override
        public void setUpdateListener(Consumer<String> listener) {}

        @Override
        public String load() {
            return converted.flexlbConfig();
        }

        @Override
        public String loadModelServiceConfig() {
            return converted.modelServiceConfig();
        }

        @Override
        public NormalizedConfig loadConfig() {
            return converted;
        }
    }
}
