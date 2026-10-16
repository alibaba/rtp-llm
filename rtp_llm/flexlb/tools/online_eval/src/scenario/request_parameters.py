"""Request payload contract shared by stage compilation and frozen decoding."""
from scenario.validation import fail, mapping, number


def normalize_request_parameters(params, loc):
    mapping(
        params,
        loc + ".params",
        {
            "input_len",
            "output_len",
            "count",
            "consume",
            "block_keys",
            "priority",
            "qos_level",
            "schedule_timeout_s",
            "stream_timeout_s",
            "post_issue_delay_s",
        },
    )
    for key, default in (("input_len", 2048), ("output_len", 10), ("count", 1)):
        params[key] = number(
            params.get(key, default),
            loc + ".params." + key,
            minimum=1,
            integer=True,
        )
    params.setdefault("consume", "immediate")
    if params["consume"] not in ("immediate", "deferred"):
        fail(loc, "consume must be immediate or deferred")
    if "post_issue_delay_s" in params:
        pause = number(
            params["post_issue_delay_s"], loc + ".params.post_issue_delay_s"
        )
        if pause > 2:
            fail(loc, "post_issue_delay_s cannot exceed two seconds")
    if "block_keys" in params:
        keys = params["block_keys"]
        if (
            not isinstance(keys, list)
            or not 1 <= len(keys) <= 4096
            or any(
                type(k) is not int or not -(2**63) <= k < 2**63 for k in keys
            )
        ):
            fail(loc, "block_keys must be 1..4096 explicit int64 keys")
    for key in ("priority", "qos_level"):
        if key in params and (
            type(params[key]) is not int or not -(2**31) <= params[key] < 2**31
        ):
            fail(loc, f"{key} must be an explicit int32 protocol value")
    for key in ("schedule_timeout_s", "stream_timeout_s"):
        if key in params:
            params[key] = number(
                params[key], loc + ".params." + key, minimum=0.001
            )
            if params[key] > 60:
                fail(loc, f"{key} cannot exceed 60 seconds")
