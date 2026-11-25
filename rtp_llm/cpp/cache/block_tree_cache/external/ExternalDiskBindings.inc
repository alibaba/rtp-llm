// C API for the native Disk coordinator. Views borrow memory from their opaque
// result handle until the matching destroy call.

int btc_tree_configure_disk(void* tree, size_t capacity, size_t ranks, bool trace) {
    return guard(
        [&] {
            static_cast<ExternalBlockTreeDriver*>(tree)->configureDisk(capacity, ranks, trace);
            return 0;
        },
        -1);
}

int btc_tree_pin_disk_load(void* tree, const DiskResidency* pages, size_t count, uint64_t* lease) {
    return guard(
        [&] {
            *lease = static_cast<ExternalBlockTreeDriver*>(tree)->pinDiskLoad(pages, count);
            return 0;
        },
        -1);
}

int btc_tree_complete_disk_load(void* tree, uint64_t lease, bool success) {
    return guard(
        [&] {
            static_cast<ExternalBlockTreeDriver*>(tree)->completeDiskLoad(lease, success);
            return 0;
        },
        -1);
}

int btc_tree_retry_disk_writes(void* tree, const DiskResidency* host_pages, size_t count, const Key* request) {
    return guard(
        [&] {
            static_cast<ExternalBlockTreeDriver*>(tree)->retryDiskWrites(host_pages, count, request->view());
            return 0;
        },
        -1);
}

int btc_tree_take_disk_jobs(void* tree, ExternalDiskJobsView* view, void** result) {
    return guard(
        [&] {
            auto pending = std::make_unique<ExternalDiskJobs>(
                static_cast<ExternalBlockTreeDriver*>(tree)->takeDiskJobs());
            *view = pending->view();
            *result = pending.release();
            return 0;
        },
        -1);
}

void btc_tree_destroy_disk_jobs(void* result) {
    delete static_cast<ExternalDiskJobs*>(result);
}

int btc_tree_settle_disk_job(void*                        tree,
                             uint64_t                    id,
                             size_t                      acknowledgements,
                             bool                        failure,
                             ExternalDiskSettlementView* view,
                             void**                      result) {
    return guard(
        [&] {
            auto settlement = std::make_unique<ExternalDiskSettlement>(
                static_cast<ExternalBlockTreeDriver*>(tree)->settleDiskJob(id, acknowledgements, failure));
            *view = settlement->view();
            *result = settlement.release();
            return 0;
        },
        -1);
}

void btc_tree_destroy_disk_settlement(void* result) {
    delete static_cast<ExternalDiskSettlement*>(result);
}

int btc_tree_disk_has_pending_work(void* tree) {
    return guard([&] { return static_cast<ExternalBlockTreeDriver*>(tree)->diskHasPendingWork() ? 1 : 0; }, -1);
}

int btc_tree_disk_reset_blocked(void* tree) {
    return guard([&] { return static_cast<ExternalBlockTreeDriver*>(tree)->diskResetBlocked() ? 1 : 0; }, -1);
}

int btc_tree_reset_disk(void* tree) {
    return guard(
        [&] {
            static_cast<ExternalBlockTreeDriver*>(tree)->resetDisk();
            return 0;
        },
        -1);
}
