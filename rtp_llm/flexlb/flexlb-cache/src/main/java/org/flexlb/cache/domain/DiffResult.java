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
     * Check if there are changes
     */
    public boolean hasChanges() {
        return !addedBlocks.isEmpty() || !removedBlocks.isEmpty();
    }

    /**
     * Create empty diff result
     */
    public static DiffResult empty() {
        return DiffResult.builder()
                .addedBlocks(Collections.emptySet())
                .removedBlocks(Collections.emptySet())
                .build();
    }
}
