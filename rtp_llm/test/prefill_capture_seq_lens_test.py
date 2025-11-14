import argparse
import os
import tempfile
from unittest import TestCase, main

from rtp_llm.server.server_args.hw_kernel_group_args import (
    _parse_decode_capture_config,
    _parse_prefill_capture_config,
)


class PrefillCaptureSeqLensTest(TestCase):
    """Test _parse_prefill_capture_config function"""

    def test_comma_separated_list(self):
        """Test comma-separated list format"""
        result = _parse_prefill_capture_config("10,100,500,1000,2000")
        self.assertEqual(result, [10, 100, 500, 1000, 2000])

    def test_comma_separated_list_with_spaces(self):
        """Test comma-separated list with spaces"""
        result = _parse_prefill_capture_config("10, 100, 500, 1000, 2000")
        self.assertEqual(result, [10, 100, 500, 1000, 2000])

    def test_comma_separated_list_filters_invalid(self):
        """Test that invalid values (non-positive) are filtered out"""
        result = _parse_prefill_capture_config("10,0,-5,100,500")
        self.assertEqual(result, [10, 100, 500])

    def test_range_format(self):
        """Test range format (max:step)"""
        result = _parse_prefill_capture_config("100:10")
        expected = list(range(10, 101, 10))
        if 100 not in expected:
            expected.append(100)
        self.assertEqual(result, expected)

    def test_range_format_large(self):
        """Test range format with large values"""
        result = _parse_prefill_capture_config("16384:128")
        expected = list(range(128, 16385, 128))
        if 16384 not in expected:
            expected.append(16384)
        self.assertEqual(result, expected)

    def test_range_format_exact_match(self):
        """Test range format where max is already in the range"""
        result = _parse_prefill_capture_config("100:50")
        # Should be [50, 100]
        self.assertEqual(result, [50, 100])

    def test_file_path_absolute(self):
        """Test file path format with absolute path"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("10\n")
            f.write("100\n")
            f.write("500\n")
            f.write("1000\n")
            f.write("2000\n")
            temp_file = f.name

        try:
            result = _parse_prefill_capture_config(temp_file)
            self.assertEqual(result, [10, 100, 500, 1000, 2000])
        finally:
            os.unlink(temp_file)

    def test_file_path_file_protocol(self):
        """Test file path format with file:// protocol"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("10\n")
            f.write("100\n")
            f.write("500\n")
            temp_file = f.name

        try:
            result = _parse_prefill_capture_config(f"file://{temp_file}")
            self.assertEqual(result, [10, 100, 500])
        finally:
            os.unlink(temp_file)

    def test_file_path_with_comments(self):
        """Test file path format with comments and empty lines"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("# This is a comment\n")
            f.write("10\n")
            f.write("\n")  # Empty line
            f.write("100\n")
            f.write("# Another comment\n")
            f.write("500\n")
            temp_file = f.name

        try:
            result = _parse_prefill_capture_config(temp_file)
            self.assertEqual(result, [10, 100, 500])
        finally:
            os.unlink(temp_file)

    def test_file_path_filters_invalid_lines(self):
        """Test that invalid lines in file are filtered out"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("10\n")
            f.write("invalid\n")
            f.write("100\n")
            f.write("0\n")  # Non-positive
            f.write("500\n")
            temp_file = f.name

        try:
            result = _parse_prefill_capture_config(temp_file)
            self.assertEqual(result, [10, 100, 500])
        finally:
            os.unlink(temp_file)

    def test_empty_config_raises_error(self):
        """Test that empty config raises ArgumentTypeError"""
        with self.assertRaises(argparse.ArgumentTypeError) as context:
            _parse_prefill_capture_config("")
        self.assertIn("prefill_capture_config must be set", str(context.exception))

    def test_empty_string_config_raises_error(self):
        """Test that empty string config raises ArgumentTypeError"""
        with self.assertRaises(argparse.ArgumentTypeError) as context:
            _parse_prefill_capture_config("")
        self.assertIn("prefill_capture_config must be set", str(context.exception))

    def test_file_not_found_raises_error(self):
        """Test that non-existent file raises ArgumentTypeError"""
        with self.assertRaises(argparse.ArgumentTypeError) as context:
            _parse_prefill_capture_config("/nonexistent/file/path.txt")
        self.assertIn("Prefill capture file not found", str(context.exception))

    def test_invalid_range_format_raises_error(self):
        """Test that invalid range format raises ArgumentTypeError"""
        with self.assertRaises(argparse.ArgumentTypeError) as context:
            _parse_prefill_capture_config("100:10:5")  # Too many colons
        self.assertIn("Range format must be 'max:step'", str(context.exception))

    def test_invalid_range_negative_values_raises_error(self):
        """Test that negative range values raise ArgumentTypeError"""
        with self.assertRaises(argparse.ArgumentTypeError) as context:
            _parse_prefill_capture_config("-100:10")
        self.assertIn(
            "max_seq_len and step must be positive integers", str(context.exception)
        )

    def test_invalid_range_zero_step_raises_error(self):
        """Test that zero step raises ArgumentTypeError"""
        with self.assertRaises(argparse.ArgumentTypeError) as context:
            _parse_prefill_capture_config("100:0")
        self.assertIn(
            "max_seq_len and step must be positive integers", str(context.exception)
        )

    def test_empty_comma_list_raises_error(self):
        """Test that empty comma-separated list raises ArgumentTypeError"""
        with self.assertRaises(argparse.ArgumentTypeError) as context:
            _parse_prefill_capture_config(",,,")  # Only commas
        self.assertIn(
            "prefill_capture_config contains no valid sequence lengths",
            str(context.exception),
        )


class DecodeCaptureBatchSizesTest(TestCase):
    """Test _parse_decode_capture_config function"""

    def test_comma_separated_list(self):
        """Test comma-separated list format"""
        result = _parse_decode_capture_config("1,2,4,8,16,32")
        self.assertEqual(result, [1, 2, 4, 8, 16, 32])

    def test_comma_separated_list_with_spaces(self):
        """Test comma-separated list with spaces"""
        result = _parse_decode_capture_config("1, 2, 4, 8, 16, 32")
        self.assertEqual(result, [1, 2, 4, 8, 16, 32])

    def test_comma_separated_list_filters_invalid(self):
        """Test that invalid values (non-positive) are filtered out"""
        result = _parse_decode_capture_config("1,0,-5,2,4")
        self.assertEqual(result, [1, 2, 4])

    def test_empty_config_returns_empty_list(self):
        """Test that empty config returns empty list"""
        result = _parse_decode_capture_config("")
        self.assertEqual(result, [])

    def test_empty_string_config_returns_empty_list(self):
        """Test that empty string config returns empty list"""
        result = _parse_decode_capture_config("   ")
        self.assertEqual(result, [])

    def test_empty_comma_list_returns_empty_list(self):
        """Test that empty comma-separated list returns empty list (no error)"""
        result = _parse_decode_capture_config(",,,")
        self.assertEqual(result, [])


if __name__ == "__main__":
    main()
