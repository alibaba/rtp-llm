package org.flexlb.dao.route;

/**
 * Distinguishes decisions for one business request ID. Generation covers
 * Prefill, Decode, and PDFusion; Encoder is the preceding EPD phase.
 */
public enum RequestPhase {
    ENCODER,
    GENERATION
}
