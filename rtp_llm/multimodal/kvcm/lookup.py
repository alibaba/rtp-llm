"""Shared metadata discovery helpers for the native object client."""


def _cached_vipserver_resolver(domain):
    # Config initialization has already seeded RTP's existing refresh machinery.
    from rtp_llm.vipserver import get_host_list_by_domain

    return get_host_list_by_domain(domain)


class KvMetaLookupError(RuntimeError):
    def __init__(self, message, *, reason="discovery", code=None):
        super().__init__(message)
        self.reason = reason
        self.code = code
        self.data_transfer_may_be_inflight = False


class KvMetaLookupTimeout(TimeoutError):
    data_transfer_may_be_inflight = False
