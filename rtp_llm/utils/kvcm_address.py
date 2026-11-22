def normalize_kvcm_address(server_address: str) -> tuple[str, str]:
    address = server_address.strip()
    vipserver_prefix = "vipserver://"
    if address.startswith(vipserver_prefix) and address != vipserver_prefix:
        return "", address[len(vipserver_prefix) :]
    if address.endswith(".vipserver") and ":" not in address and "/" not in address:
        return "", address
    return server_address, ""
