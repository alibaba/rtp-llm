# type: ignore
import sys
from io import StringIO
from unittest import SkipTest, TestCase, main

import torch

from rtp_llm.models_py.utils.debug import cudagraph_debug_kernel


class DebugTest(TestCase):

    def setUp(self) -> None:
        """Setup test environment."""
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def test_cudagraph_debug_kernel_2d(self) -> None:
        """Test debug kernel with 2D tensor."""
        print("Testing cudagraph_debug_kernel with 2D tensor:", flush=True)

        # Create a 2D test tensor
        data = torch.randn((10, 20), dtype=torch.float32, device="cuda")

        # Capture stdout to verify print output
        captured_output = StringIO()
        sys.stdout = captured_output

        try:
            # Call the debug kernel
            cudagraph_debug_kernel(
                data=data, info_id=1, m=5, n=10, row_len=20, name="test_2d_tensor"
            )

            # Get output
            output = captured_output.getvalue()

            # Verify output contains shape info
            self.assertIn("test_2d_tensor shape is torch.Size([10, 20])", output)

        finally:
            # Restore stdout
            sys.stdout = sys.__stdout__
            print(f"Test output: {captured_output.getvalue()}", flush=True)

        print("2D tensor test passed", flush=True)

    def test_cudagraph_debug_kernel_1d(self) -> None:
        """Test debug kernel with 1D tensor (should be unsqueezed to 2D)."""
        print("Testing cudagraph_debug_kernel with 1D tensor:", flush=True)

        # Create a 1D test tensor
        data = torch.randn(50, dtype=torch.float32, device="cuda")

        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output

        try:
            # Call the debug kernel
            cudagraph_debug_kernel(
                data=data, info_id=2, m=1, n=50, name="test_1d_tensor"
            )

            output = captured_output.getvalue()
            self.assertIn("test_1d_tensor shape is torch.Size([50])", output)

        finally:
            sys.stdout = sys.__stdout__
            print(f"Test output: {captured_output.getvalue()}", flush=True)

        print("1D tensor test passed", flush=True)

    def test_cudagraph_debug_kernel_default_params(self) -> None:
        """Test debug kernel with default parameters."""
        print("Testing cudagraph_debug_kernel with default params:", flush=True)

        # Create a test tensor
        data = torch.randn((16, 32), dtype=torch.float32, device="cuda")

        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output

        try:
            # Call with default parameters
            cudagraph_debug_kernel(data=data)

            output = captured_output.getvalue()
            self.assertIn(
                "cudagraph_debug_kernel shape is torch.Size([16, 32])", output
            )

        finally:
            sys.stdout = sys.__stdout__
            print(f"Test output: {captured_output.getvalue()}", flush=True)

        print("Default params test passed", flush=True)

    def test_cudagraph_debug_kernel_large_tensor(self) -> None:
        """Test debug kernel with large tensor (partial print)."""
        print("Testing cudagraph_debug_kernel with large tensor:", flush=True)

        # Create a large tensor
        data = torch.randn((100, 200), dtype=torch.float32, device="cuda")

        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output

        try:
            # Call with limited rows/cols
            cudagraph_debug_kernel(
                data=data,
                info_id=3,
                m=10,  # Only print 10 rows
                n=20,  # Only print 20 cols
                name="test_large_tensor",
            )

            output = captured_output.getvalue()
            self.assertIn("test_large_tensor shape is torch.Size([100, 200])", output)

        finally:
            sys.stdout = sys.__stdout__
            print(f"Test output: {captured_output.getvalue()}", flush=True)

        print("Large tensor test passed", flush=True)

    def test_cudagraph_debug_kernel_different_dtypes(self) -> None:
        """Test debug kernel converts different dtypes to float32."""
        print("Testing cudagraph_debug_kernel with different dtypes:", flush=True)

        # Test with different dtypes
        dtypes = [torch.float16, torch.bfloat16, torch.float32]

        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                data = torch.randn((8, 16), dtype=dtype, device="cuda")

                # Capture stdout
                captured_output = StringIO()
                sys.stdout = captured_output

                try:
                    cudagraph_debug_kernel(
                        data=data, info_id=4, m=4, n=8, name=f"test_{dtype}"
                    )

                    output = captured_output.getvalue()
                    self.assertIn(f"test_{dtype} shape is torch.Size([8, 16])", output)

                finally:
                    sys.stdout = sys.__stdout__

                print(f"{dtype} test passed", flush=True)

        print("Different dtypes test passed", flush=True)

    def test_cudagraph_debug_kernel_non_contiguous(self) -> None:
        """Test debug kernel with non-contiguous tensor."""
        print("Testing cudagraph_debug_kernel with non-contiguous tensor:", flush=True)

        # Create a non-contiguous tensor
        data = torch.randn((20, 40), dtype=torch.float32, device="cuda")
        non_contiguous_data = data.t()  # Transpose makes it non-contiguous

        self.assertFalse(non_contiguous_data.is_contiguous())

        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output

        try:
            # Should handle non-contiguous tensor by making it contiguous
            cudagraph_debug_kernel(
                data=non_contiguous_data,
                info_id=5,
                m=10,
                n=10,
                name="test_non_contiguous",
            )

            output = captured_output.getvalue()
            self.assertIn("test_non_contiguous shape is torch.Size([40, 20])", output)

        finally:
            sys.stdout = sys.__stdout__
            print(f"Test output: {captured_output.getvalue()}", flush=True)

        print("Non-contiguous tensor test passed", flush=True)

    def test_cudagraph_debug_kernel_edge_cases(self) -> None:
        """Test debug kernel with edge cases."""
        print("Testing cudagraph_debug_kernel with edge cases:", flush=True)

        # Test with very small tensor
        small_data = torch.randn((1, 1), dtype=torch.float32, device="cuda")

        captured_output = StringIO()
        sys.stdout = captured_output

        try:
            cudagraph_debug_kernel(data=small_data, info_id=6, name="test_small")

            output = captured_output.getvalue()
            self.assertIn("test_small shape is torch.Size([1, 1])", output)

        finally:
            sys.stdout = sys.__stdout__
            print(f"Test output: {captured_output.getvalue()}", flush=True)

        print("Edge cases test passed", flush=True)


if __name__ == "__main__":
    main()
