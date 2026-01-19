"""Request ID generator module.

This module provides functionality to generate unique request IDs using a
snowflake-like algorithm that combines timestamp, machine ID, and sequence number.
"""

import hashlib
import time


def generate_request_id(
    ip: str,
    port: int,
    server_id: str,
    sequence: int,
) -> int:
    """Generate a unique request ID.

    The request ID is a 64-bit integer composed of:
    - 40 bits: relative timestamp (milliseconds since 2020-01-01)
    - 12 bits: machine ID (derived from ip, port, and server_id)
    - 12 bits: sequence number

    Args:
        ip: IP address of the frontend server
        port: Port number of the frontend server
        server_id: Server ID string
        sequence: Sequence number (should be incremented for each request)

    Returns:
        A unique 64-bit integer request ID
    """
    # This allows us to use fewer bits while maintaining millisecond precision
    EPOCH_2020_MS = 1577836800000  # 2020-01-01 00:00:00 UTC in milliseconds
    current_timestamp_ms = int(time.time() * 1000)
    relative_timestamp = (
        current_timestamp_ms - EPOCH_2020_MS
    ) & 0xFFFFFFFFFF  # 40 bits mask

    # Generate machine_id by hashing ip, port, and server_id
    # Use SHA256 for better hash distribution, then take 12 bits to fit within int64
    # 12 bits gives us 4,096 possible machine values
    server_id_str = str(server_id)
    # Create a hash from ip, port, and server_id using SHA256 for better distribution
    hash_input = f"{ip}:{port}:{server_id_str}".encode("utf-8")
    hash_value = int(hashlib.sha256(hash_input).hexdigest(), 16)
    # Use 12 bits of the hash as machine_id (4,096 possible values)
    machine_id = hash_value & 0xFFF  # 12 bits
    sequence_value = sequence % 4096  # 12 bits

    # Redesigned format to fit within int64 (64 bits total):
    # relative_timestamp (40 bits, high) | machine_id (12 bits, middle) | sequence (12 bits, low)
    # This ensures no overflow while maintaining good uniqueness guarantees
    # 40 bits for relative timestamp supports ~35 years from 2020 (2^40 ms ≈ 35 years)
    # 12 bits for machine_id supports 4,096 different machines
    # 12 bits for sequence supports 4,096 requests per millisecond per machine
    request_id = (relative_timestamp << 24) | (machine_id << 12) | sequence_value
    return request_id
