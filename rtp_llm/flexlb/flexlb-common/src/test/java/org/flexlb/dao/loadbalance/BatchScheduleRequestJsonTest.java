package org.flexlb.dao.loadbalance;

import org.flexlb.util.JsonUtils;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Wire-compatibility coverage for the additive allocation-dimension flags. */
class BatchScheduleRequestJsonTest {

    @Test
    void legacyBodyStillRequestsBothAssignments() {
        BatchScheduleRequest request = JsonUtils.toObject(
                "{\"batch_count\":2}", BatchScheduleRequest.class);

        assertTrue(request.isAssignBe());
        assertTrue(request.isAssignFe());
    }

    @Test
    void feOnlyFlagsRoundTripUnderSnakeCaseKeys() {
        BatchScheduleRequest request = JsonUtils.toObject(
                "{\"batch_count\":2,\"assign_be\":false,\"assign_fe\":true}",
                BatchScheduleRequest.class);

        assertFalse(request.isAssignBe());
        assertTrue(request.isAssignFe());
        String json = JsonUtils.toString(request);
        assertTrue(json.contains("\"assign_be\":false"));
        assertTrue(json.contains("\"assign_fe\":true"));
    }
}
