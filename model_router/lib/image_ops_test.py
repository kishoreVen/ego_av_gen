
import base64
import io
import unittest

from PIL import Image

from model_router.lib.media import compress_for_reference


def _make_image(width: int, height: int, mode: str = "RGB") -> Image.Image:
    """Create a solid-color test image."""
    color = (255, 0, 0) if mode == "RGB" else (255, 0, 0, 128)
    return Image.new(mode, (width, height), color)


def _decode_data_uri(data_uri: str) -> Image.Image:
    """Decode a data-URI base64 string back to a PIL Image."""
    header, b64 = data_uri.split(",", 1)
    return Image.open(io.BytesIO(base64.b64decode(b64)))


class TestCompressForReference(unittest.TestCase):
    def test_small_image_not_resized(self):
        img = _make_image(512, 512)
        result = compress_for_reference([img])
        self.assertEqual(len(result), 1)
        decoded = _decode_data_uri(result[0])
        self.assertEqual(decoded.size, (512, 512))

    def test_large_image_downscaled(self):
        img = _make_image(2048, 1024)
        result = compress_for_reference([img], max_px=1024)
        self.assertEqual(len(result), 1)
        decoded = _decode_data_uri(result[0])
        self.assertLessEqual(max(decoded.size), 1024)

    def test_custom_max_px(self):
        img = _make_image(800, 600)
        result = compress_for_reference([img], max_px=400)
        decoded = _decode_data_uri(result[0])
        self.assertLessEqual(max(decoded.size), 400)

    def test_rgba_converted_to_rgb(self):
        img = _make_image(100, 100, mode="RGBA")
        result = compress_for_reference([img])
        decoded = _decode_data_uri(result[0])
        self.assertEqual(decoded.mode, "RGB")

    def test_grayscale_converted_to_rgb(self):
        img = Image.new("L", (100, 100), 128)
        result = compress_for_reference([img])
        decoded = _decode_data_uri(result[0])
        self.assertEqual(decoded.mode, "RGB")

    def test_string_passthrough(self):
        uri = "data:image/png;base64,abc123"
        result = compress_for_reference([uri])
        self.assertEqual(result, [uri])

    def test_mixed_inputs(self):
        img = _make_image(100, 100)
        uri = "data:image/png;base64,abc123"
        result = compress_for_reference([img, uri])
        self.assertEqual(len(result), 2)
        self.assertTrue(result[0].startswith("data:image/jpeg;base64,"))
        self.assertEqual(result[1], uri)

    def test_empty_list(self):
        self.assertEqual(compress_for_reference([]), [])

    def test_output_is_jpeg_data_uri(self):
        img = _make_image(100, 100)
        result = compress_for_reference([img])
        self.assertTrue(result[0].startswith("data:image/jpeg;base64,"))

    def test_aspect_ratio_preserved(self):
        img = _make_image(2000, 1000)
        result = compress_for_reference([img], max_px=500)
        decoded = _decode_data_uri(result[0])
        w, h = decoded.size
        self.assertAlmostEqual(w / h, 2.0, delta=0.05)


if __name__ == "__main__":
    unittest.main()
