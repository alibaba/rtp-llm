package org.flexlb.balance.resource;

import org.flexlb.enums.ResourceMeasureIndicatorEnum;
import org.springframework.stereotype.Component;

import java.util.EnumMap;
import java.util.List;
import java.util.Map;

/**
 * Resource measure factory
 * Retrieves appropriate resource measure based on RoleType
 *
 * @author saichen.sm
 * @since 2025/12/23
 */
@Component
public class ResourceMeasureFactory {

    private final Map<ResourceMeasureIndicatorEnum, ResourceMeasure> measureMap;

    public ResourceMeasureFactory(List<ResourceMeasure> measureList) {
        this.measureMap = new EnumMap<>(ResourceMeasureIndicatorEnum.class);
        for (ResourceMeasure measure : measureList) {
            measureMap.put(measure.getResourceMeasureIndicator(), measure);
        }
    }

    /**
     * Get resource measure based on resource indicator
     *
     * @param measureIndicator Resource measure indicator
     * @return Resource measure instance, or null if not found
     */
    public ResourceMeasure getMeasure(ResourceMeasureIndicatorEnum measureIndicator) {
        return measureMap.get(measureIndicator);
    }
}
