"""Worker freshness and generation identity shared by acquisition and analysis."""

def healthy(row, maximum):
    # Required observations never fall back to a different source or default true.
    poll, applied = row["status_age_ms"], row["applied_status_age_ms"]
    return (row["reported_alive"] is True and row["lifecycle_state"] == "active"
            and poll is not None and applied is not None
            and 0 <= poll <= maximum and 0 <= applied <= maximum)


def collateral(rows, baseline, victims, maximum):
    """Absence, generation replacement and stale status all affect healthy survivors."""
    return sorted(address for address, old in baseline.items() if address not in victims
                  and (address not in rows or rows[address]["generation"] != old["generation"]
                       or not healthy(rows[address], maximum)))


def converged(rows, names, keys, versions, maximum):
    if set(rows) != set(names):
        return False
    return all(healthy(row, maximum) and (row["role"] != "prefill" or (
        row.get("cache_indexed") is True and row.get("cache_key_size") == keys
        and row.get("cache_version") == versions[address]
        and row.get("cache_indexed_version") == versions[address]))
               for address, row in rows.items())
