package org.flexlb.enums;

import lombok.Getter;

/**
 * FlexLB-local lifecycle state for a request after route selection.
 */
@Getter
public enum TaskStateEnum {

    CREATED("created"),
    IN_TRANSIT("in_transit"),
    CONFIRMED("confirmed"),
    RUNNING("running"),
    LOST("lost"),
    FINISHED("finished"),
    CLEANED("timeout_cleaned");

    private final String value;

    TaskStateEnum(String value) {
        this.value = value;
    }
}
