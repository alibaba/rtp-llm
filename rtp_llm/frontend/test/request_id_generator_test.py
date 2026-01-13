"""Test for request_id_generator module.

This test simulates 4 different frontend servers generating 10000 request IDs each
and checks for duplicates.
"""

import collections
import logging
import threading
import time
from unittest import TestCase, main

from rtp_llm.frontend.request_id_generator import generate_request_id


class RequestIdGeneratorTest(TestCase):
    """Test cases for request ID generator."""

    def test_no_duplicates_across_multiple_frontends(self):
        """Test that 4 different frontends can generate 10000 unique IDs each without duplicates."""
        num_frontends = 4
        num_requests_per_frontend = 10000

        # Simulate 4 different frontend servers with different IPs, ports, and server IDs
        frontend_configs = [
            {"ip": "192.168.1.1", "port": 8000, "server_id": "0"},
            {"ip": "192.168.1.2", "port": 8001, "server_id": "1"},
            {"ip": "192.168.1.3", "port": 8002, "server_id": "2"},
            {"ip": "10.0.0.1", "port": 9000, "server_id": "3"},
        ]

        # Store all generated request IDs
        all_request_ids = set()
        request_ids_by_frontend = []

        # Statistics: track IDs generated per second
        # Key: timestamp (seconds), Value: list of (request_id, frontend_idx)
        ids_by_second = collections.defaultdict(list)

        # Generate request IDs for each frontend
        for frontend_idx, config in enumerate(frontend_configs):
            frontend_ids = []
            for sequence in range(num_requests_per_frontend):
                # Record timestamp before generating ID
                timestamp_sec = int(time.time())

                request_id = generate_request_id(
                    ip=config["ip"],
                    port=config["port"],
                    server_id=config["server_id"],
                    sequence=sequence,
                )

                frontend_ids.append(request_id)
                all_request_ids.add(request_id)

                # Record ID with its timestamp and frontend
                ids_by_second[timestamp_sec].append((request_id, frontend_idx))
            request_ids_by_frontend.append(frontend_ids)

        # Check for duplicates within each frontend
        for frontend_idx, frontend_ids in enumerate(request_ids_by_frontend):
            unique_ids = set(frontend_ids)
            self.assertEqual(
                len(unique_ids),
                len(frontend_ids),
                f"Frontend {frontend_idx} generated duplicate IDs. "
                f"Expected {len(frontend_ids)} unique IDs, got {len(unique_ids)}",
            )

        # Check for duplicates across all frontends
        total_expected = num_frontends * num_requests_per_frontend
        self.assertEqual(
            len(all_request_ids),
            total_expected,
            f"Found duplicate IDs across frontends. "
            f"Expected {total_expected} unique IDs, got {len(all_request_ids)}",
        )

        # Statistics: Find the second with the most IDs generated
        max_ids_per_second = 0
        max_second = None
        for timestamp_sec, id_list in ids_by_second.items():
            if len(id_list) > max_ids_per_second:
                max_ids_per_second = len(id_list)
                max_second = timestamp_sec

        # Check for duplicates in the second with most IDs
        if max_second is not None:
            ids_in_max_second = [req_id for req_id, _ in ids_by_second[max_second]]
            unique_ids_in_max_second = set(ids_in_max_second)
            has_duplicates = len(unique_ids_in_max_second) < len(ids_in_max_second)

            # Count IDs per frontend in that second
            frontend_counts = collections.defaultdict(int)
            for _, frontend_idx in ids_by_second[max_second]:
                frontend_counts[frontend_idx] += 1

        logging.info(f"✓ Successfully generated {total_expected} unique request IDs")
        logging.info(f"  - {num_frontends} frontends")
        logging.info(f"  - {num_requests_per_frontend} requests per frontend")
        logging.info(f"  - No duplicates found")
        logging.info(f"\n📊 Statistics:")
        logging.info(f"  - Total seconds with ID generation: {len(ids_by_second)}")
        if max_second is not None:
            logging.info(f"  - Maximum IDs generated in 1 second: {max_ids_per_second}")
            logging.info(f"  - Timestamp of max second: {max_second}")
            logging.info(
                f"  - Duplicates in max second: {'Yes' if has_duplicates else 'No'}"
            )
            logging.info(
                f"  - Unique IDs in max second: {len(unique_ids_in_max_second)}"
            )
            logging.info(f"  - Frontend distribution in max second:")
            for frontend_idx in sorted(frontend_counts.keys()):
                logging.info(
                    f"      Frontend {frontend_idx}: {frontend_counts[frontend_idx]} IDs"
                )

    def test_concurrent_generation(self):
        """Test concurrent request ID generation from multiple frontends."""
        num_frontends = 4
        num_requests_per_frontend = 10000

        frontend_configs = [
            {"ip": "192.168.1.1", "port": 8000, "server_id": "0"},
            {"ip": "192.168.1.1", "port": 8010, "server_id": "1"},
            {"ip": "192.168.1.1", "port": 8020, "server_id": "2"},
            {"ip": "192.168.1.1", "port": 8030, "server_id": "3"},
        ]

        all_request_ids = set()
        lock = threading.Lock()

        def generate_ids_for_frontend(config, frontend_idx):
            """Generate IDs for a single frontend."""
            frontend_ids = []
            for sequence in range(num_requests_per_frontend):
                request_id = generate_request_id(
                    ip=config["ip"],
                    port=config["port"],
                    server_id=config["server_id"],
                    sequence=sequence,
                )
                frontend_ids.append(request_id)
                with lock:
                    all_request_ids.add(request_id)

        # Create threads for each frontend
        threads = []
        for frontend_idx, config in enumerate(frontend_configs):
            thread = threading.Thread(
                target=generate_ids_for_frontend, args=(config, frontend_idx)
            )
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check for duplicates
        total_expected = num_frontends * num_requests_per_frontend
        self.assertEqual(
            len(all_request_ids),
            total_expected,
            f"Found duplicate IDs in concurrent generation. "
            f"Expected {total_expected} unique IDs, got {len(all_request_ids)}",
        )

        logging.info(
            f"✓ Successfully generated {total_expected} unique request IDs concurrently"
        )
        logging.info(f"  - {num_frontends} frontends running in parallel")
        logging.info(f"  - {num_requests_per_frontend} requests per frontend")
        logging.info(f"  - No duplicates found")

    def test_request_id_format(self):
        """Test that generated request IDs are valid 64-bit integers."""
        request_id = generate_request_id(
            ip="192.168.1.1",
            port=8000,
            server_id="0",
            sequence=1,
        )

        # Check that it's an integer
        self.assertIsInstance(request_id, int)

        # Check that it fits in 64 bits (should be positive and less than 2^63)
        self.assertGreater(request_id, 0)
        self.assertLess(request_id, 2**63)

        logging.info(f"✓ Generated request ID: {request_id}")
        logging.info(f"  - Type: {type(request_id)}")
        logging.info(f"  - Value fits in 64 bits: {request_id < 2**63}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(filename)s %(funcName)s %(lineno)d %(levelname)s %(asctime)s %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    main()
