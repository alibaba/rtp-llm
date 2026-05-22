package org.flexlb.dispatcher;

/**
 * Wire-format constants for the FE batch-infer protocol used by the dispatcher and the FE.
 * Keep one source of truth so a future rename does not have to be hunted across files.
 */
final class DispatchProtocol {

    private DispatchProtocol() {
    }

    static final String PATH_BATCH_INFER = "/batch_infer";

    static final String FIELD_PROMPT_BATCH = "prompt_batch";

    static final String FIELD_RESPONSE_BATCH = "response_batch";

    static final String FIELD_GENERATE_CONFIG = "generate_config";
}
