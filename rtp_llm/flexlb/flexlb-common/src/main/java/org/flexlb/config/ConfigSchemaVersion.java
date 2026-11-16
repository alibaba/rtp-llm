package org.flexlb.config;

/**
 * Names the supported configuration schema versions and the current standard.
 */
public final class ConfigSchemaVersion {

    /** Historical combined FlexLB and model-service document; also used when no version is declared. */
    public static final int V0_COMPATIBILITY = 0;

    /** Current normalized FlexLB configuration document. */
    public static final int STANDARD = FlexlbConfig.CURRENT_SCHEMA_VERSION;

    private ConfigSchemaVersion() {}
}
