"""A lane can be reused only after its previous owner released resources."""


def cleanup_failure(row):
    cleanup = row.get("cleanup")
    if not isinstance(cleanup, list):
        return "missing cleanup results"
    if any(not isinstance(item, dict) or item.get("status") != "PASS" or item.get("error")
           for item in cleanup):
        return "previous instance cleanup did not complete successfully"
    if not row.get("stages"):
        return "previous instance has no verified execution result: " + str(row.get("error"))
    return None


def occupied_ports(intervals, probe):
    return [port for interval in intervals
            for port in range(interval["first"], interval["last"] + 1) if probe(port)]
