package org.flexlb.config;

/**
 * Request ownership stages eligible for priority resource reclamation.
 */
public enum VictimStage {
    DECODE_RESERVED,
    DECODE_ENGINE_OWNED
}
