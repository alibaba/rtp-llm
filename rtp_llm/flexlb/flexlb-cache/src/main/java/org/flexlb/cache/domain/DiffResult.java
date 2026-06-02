package org.flexlb.cache.domain;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serial;
import java.io.Serializable;
import java.util.Collections;
import java.util.Set;

/**
 * Diff calculation result
 * Describes differences between two cache states
 *
 * @author FlexLB
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class DiffResult implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;
    
    /**
     * Added cache blocks
     */
    @Builder.Default
    private Set<Long> addedBlocks = Collections.emptySet();
    
    /**
     * Removed cache blocks
     */
    @Builder.Default
    private Set<Long> removedBlocks = Collections.emptySet();
    
    /**
     * Engine IP
     */
    private String engineIp;
    
    /**
     * Version number
     */
    private String version;
    
    /**
     * Check if there are changes
     */
    public boolean hasChanges() {
        return !addedBlocks.isEmpty() || !removedBlocks.isEmpty();
    }
    
    /**
     * Create empty diff result
     */
    public static DiffResult empty(String engineIp) {
        return DiffResult.builder()
                .engineIp(engineIp)
                .addedBlocks(Collections.emptySet())
                .removedBlocks(Collections.emptySet())
                .build();
    }
}