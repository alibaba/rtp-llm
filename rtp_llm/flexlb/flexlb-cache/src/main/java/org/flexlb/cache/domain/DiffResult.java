package org.flexlb.cache.domain;

import it.unimi.dsi.fastutil.longs.LongSet;
import it.unimi.dsi.fastutil.longs.LongSets;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serial;
import java.io.Serializable;

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

    @Serial private static final long serialVersionUID = 1L;

    /**
     * Added cache blocks
     */
    @Builder.Default private LongSet addedBlocks = LongSets.EMPTY_SET;

    /**
     * Removed cache blocks
     */
    @Builder.Default private LongSet removedBlocks = LongSets.EMPTY_SET;

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
                .addedBlocks(LongSets.EMPTY_SET)
                .removedBlocks(LongSets.EMPTY_SET)
                .build();
    }
}
