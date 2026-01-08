import base64
import io
import logging
import tempfile
import unittest

from PIL import Image

from rtp_llm.multimodal.multimodal_util import get_bytes_io_from_url


class TestMultiModalUtil(unittest.TestCase):
    def test_get_bytes(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp_file:
            temp_path = tmp_file.name

            image = Image.new("RGB", (200, 200), "white")
            image.save(temp_path, format="PNG")

            self.assertTrue(
                Image.open(get_bytes_io_from_url(temp_path)).size == image.size
            )

    def test_base64(self):
        buffer = io.BytesIO()

        image = Image.new("RGB", (200, 200), "white")
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        base64_str = "data:image/png;base64," + base64.b64encode(image_bytes).decode(
            "utf-8"
        )

        self.assertTrue(
            Image.open(get_bytes_io_from_url(base64_str)).size == image.size
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
