package org.flexlb.config.loading;

/**
 * Names the supported configuration schema versions and the current standard.
 */
public final class ConfigSchemaVersion {

    /**
     * Historical combined FlexLB and model-service document; also used when no version is declared.
     */
    public static final int V0_COMPATIBILITY = 0;

    public static final int V4 = 4;

    /**
     * Current normalized FlexLB configuration document.
     */
    public static final int STANDARD = V4;

    private ConfigSchemaVersion() {}
}
