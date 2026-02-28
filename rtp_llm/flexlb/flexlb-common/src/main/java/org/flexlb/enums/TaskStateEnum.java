package org.flexlb.enums;

import lombok.Getter;

/**
 * Task lifecycle state enumeration
 */
@Getter
public enum TaskStateEnum {

    /**
     * Created state - Task just added to local queue, not yet sent to Worker
     */
    CREATED("created"),

    /**
     * In-transit state - Task sent to Worker, but Worker confirmation not yet received
     */
    IN_TRANSIT("in_transit"),

    /**
     * Confirmed state - Worker confirmed task receipt (appears in runningTaskInfo or finishedTaskList)
     */
    CONFIRMED("confirmed"),

    /**
     * Running state - Task in Worker's runningTaskInfo
     */
    RUNNING("running"),

    /**
     * Lost state - Task was previously confirmed, but now neither in runningTaskInfo nor finishedTaskList
     */
    LOST("lost"),

    /**
     * Finished state - Task appears in finishedTaskList
     */
    FINISHED("finished"),

    /**
     * Cleaned state - Task forcibly cleaned due to timeout or loss
     */
    CLEANED("timeout_cleaned");
    
    private final String value;
    
    TaskStateEnum(String value) {
        this.value = value;
    }

}