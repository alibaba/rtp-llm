package org.flexlb.balance.schedule;

/**
 * Checks scheduler assembly and startup ordering, including ownership of the shared dispatcher at shutdown.
 */
class SchedulerConfigurationTest {

    /**
     * Receives the lazily assembled scheduler to verify Spring resolves the shared request entry point.
     */
    static class RequestConsumer {
        final AbstractRequestScheduler entry;

        RequestConsumer(
                @org.springframework.context.annotation.Lazy AbstractRequestScheduler entry) {
            this.entry = entry;
        }
    }
}
