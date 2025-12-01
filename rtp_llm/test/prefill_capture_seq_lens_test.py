import os
import tempfile
from unittest import TestCase, main

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters


class PrefillCaptureSeqLensTest(TestCase):
    def setUp(self):
        """Set up test environment"""
        # Create a test instance
        self.params = GptInitModelParameters(0, 0, 0, 0, 0)
        # Clear prefill_capture_config for each test
        self.params.py_env_configs.py_hw_kernel_config.prefill_capture_config = ""

    def test_comma_separated_list(self):
        """Test comma-separated list format"""
        self.params.py_env_configs.py_hw_kernel_config.prefill_capture_config = (
            "10,100,500,1000,2000"
        )
        result = self.params._generate_prefill_capture_seq_lens()
        self.assertEqual(result, [10, 100, 500, 1000, 2000])

    def test_comma_separated_list_with_spaces(self):
        """Test comma-separated list with spaces"""
        self.params.py_env_configs.py_hw_kernel_config.prefill_capture_config = (
            "10, 100, 500, 1000, 2000"
        )
        result = self.params._generate_prefill_capture_seq_lens()
        self.assertEqual(result, [10, 100, 500, 1000, 2000])

    def test_comma_separated_list_filters_invalid(self):
        """Test that invalid values (non-positive) are filtered out"""
        self.params.py_env_configs.py_hw_kernel_config.prefill_capture_config = (
            "10,0,-5,100,500"
        )
        result = self.params._generate_prefill_capture_seq_lens()
        self.assertEqual(result, [10, 100, 500])

    def test_range_format(self):
        """Test range format (max:step)"""
        self.params.py_env_configs.py_hw_kernel_config.prefill_capture_config = "100:10"
        result = self.params._generate_prefill_capture_seq_lens()
        expected = list(range(10, 101, 10))
        if 100 not in expected:
            expected.append(100)
        self.assertEqual(result, expected)

    def test_range_format_large(self):
        """Test range format with large values"""
        self.params.py_env_configs.py_hw_kernel_config.prefill_capture_config = (
            "16384:128"
        )
        result = self.params._generate_prefill_capture_seq_lens()
        expected = list(range(128, 16385, 128))
        if 16384 not in expected:
            expected.append(16384)
        self.assertEqual(result, expected)

    def test_range_format_exact_match(self):
        """Test range format where max is already in the range"""
        self.params.py_env_configs.py_hw_kernel_config.prefill_capture_config = "100:50"
        result = self.params._generate_prefill_capture_seq_lens()
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
            self.params.py_env_configs.py_hw_kernel_config.prefill_capture_config = (
                temp_file
            )
            result = self.params._generate_prefill_capture_seq_lens()
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
            self.params.py_env_configs.py_hw_kernel_config.prefill_capture_config = (
                f"file://{temp_file}"
            )
            result = self.params._generate_prefill_capture_seq_lens()
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
            self.params.py_env_configs.py_hw_kernel_config.prefill_capture_config = (
                temp_file
            )
            result = self.params._generate_prefill_capture_seq_lens()
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
            self.params.py_env_configs.py_hw_kernel_config.prefill_capture_config = (
                temp_file
            )
            result = self.params._generate_prefill_capture_seq_lens()
            self.assertEqual(result, [10, 100, 500])
        finally:
            os.unlink(temp_file)

    def test_empty_config_raises_error(self):
        """Test that empty config raises ValueError"""
        # prefill_capture_config is not set (empty string)
        self.params.py_env_configs.py_hw_kernel_config.prefill_capture_config = ""
        with self.assertRaises(ValueError) as context:
            self.params._generate_prefill_capture_seq_lens()
        self.assertIn("prefill_capture_config must be set", str(context.exception))

    def test_empty_string_config_raises_error(self):
        """Test that empty string config raises ValueError"""
        self.params.py_env_configs.py_hw_kernel_config.prefill_capture_config = ""
        with self.assertRaises(ValueError) as context:
            self.params._generate_prefill_capture_seq_lens()
        self.assertIn("prefill_capture_config must be set", str(context.exception))

    def test_file_not_found_raises_error(self):
        """Test that non-existent file raises FileNotFoundError"""
        self.params.py_env_configs.py_hw_kernel_config.prefill_capture_config = (
            "/nonexistent/file/path.txt"
        )
        with self.assertRaises(FileNotFoundError) as context:
            self.params._generate_prefill_capture_seq_lens()
        self.assertIn("Prefill capture file not found", str(context.exception))

    def test_invalid_range_format_raises_error(self):
        """Test that invalid range format raises ValueError"""
        self.params.py_env_configs.py_hw_kernel_config.prefill_capture_config = (
            "100:10:5"  # Too many colons
        )
        with self.assertRaises(ValueError) as context:
            self.params._generate_prefill_capture_seq_lens()
        self.assertIn("Range format must be 'max:step'", str(context.exception))

    def test_invalid_range_negative_values_raises_error(self):
        """Test that negative range values raise ValueError"""
        self.params.py_env_configs.py_hw_kernel_config.prefill_capture_config = (
            "-100:10"
        )
        with self.assertRaises(ValueError) as context:
            self.params._generate_prefill_capture_seq_lens()
        self.assertIn(
            "max_seq_len and step must be positive integers", str(context.exception)
        )

    def test_invalid_range_zero_step_raises_error(self):
        """Test that zero step raises ValueError"""
        self.params.py_env_configs.py_hw_kernel_config.prefill_capture_config = "100:0"
        with self.assertRaises(ValueError) as context:
            self.params._generate_prefill_capture_seq_lens()
        self.assertIn(
            "max_seq_len and step must be positive integers", str(context.exception)
        )

    def test_empty_comma_list_raises_error(self):
        """Test that empty comma-separated list raises ValueError"""
        self.params.py_env_configs.py_hw_kernel_config.prefill_capture_config = (
            ",,,"  # Only commas
        )
        with self.assertRaises(ValueError) as context:
            self.params._generate_prefill_capture_seq_lens()
        self.assertIn(
            "prefill_capture_config contains no valid sequence lengths",
            str(context.exception),
        )


if __name__ == "__main__":
    main()
