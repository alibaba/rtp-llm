"""Per-forward sparse-attention workspace with an immutable global prefix."""

import torch


def combine_kv(shared, globals_by_req, swa):
    """Only refresh SWA while layers share the same global tensor.

    The caller consumes the result on the forward stream before the next
    layer updates it. The cache is cleared with the per-forward global state.
    Multi-request layouts retain the ordinary interleaved concatenation.
    """
    if len(globals_by_req) != 1:
        return torch.cat(
            [t for (g, _), sw in zip(globals_by_req, swa) for t in (g, sw)]
        )
    g, sw = globals_by_req[0][0], swa[0]
    # For small encoder workspaces the extra Python/slice/copy launch cost
    # exceeds the saved traffic. The bounded decoder tail always benefits;
    # long encoder groups also amortize that cost with >=64K global rows.
    if sw.shape[0] > 128 and g.shape[0] < 65536:
        return torch.cat((g, sw))
    signature = (sw.shape, sw.dtype, sw.device)
    cached = shared.get("prefill_kv_workspace")
    if cached is not None:
        source, previous_signature, output = cached
        if source is g and previous_signature == signature:
            output[g.shape[0] :].copy_(sw)
            return output
        # Drop the old buffer before allocating a new source group's layout.
        shared.pop("prefill_kv_workspace")
        del cached, source, output
    output = torch.cat((g, sw))
    shared["prefill_kv_workspace"] = (g, signature, output)
    return output
