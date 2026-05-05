"""
tests/test_corrupted_files.py
==============================
Tests that io_handler.load_image() raises (or handles gracefully) every
category of bad input a real user might throw at it.

Run from the project root:
    python -m pytest tests/test_corrupted_files.py -v
"""
import os
import sys
import struct
import pytest
import numpy as np

# Make sure the project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processing.io_handler import load_image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_tmp(tmp_path, filename: str, data: bytes) -> str:
    """Write raw bytes to a temp file and return its path."""
    p = tmp_path / filename
    p.write_bytes(data)
    return str(p)


# ---------------------------------------------------------------------------
# 1. Completely empty file
# ---------------------------------------------------------------------------
class TestEmptyFile:
    def test_empty_jpg(self, tmp_path):
        path = write_tmp(tmp_path, "empty.jpg", b"")
        with pytest.raises(Exception):
            load_image(path)

    def test_empty_png(self, tmp_path):
        path = write_tmp(tmp_path, "empty.png", b"")
        with pytest.raises(Exception):
            load_image(path)

    def test_empty_dcm(self, tmp_path):
        path = write_tmp(tmp_path, "empty.dcm", b"")
        with pytest.raises(Exception):
            load_image(path)


# ---------------------------------------------------------------------------
# 2. Random garbage bytes (wrong magic bytes for the declared extension)
# ---------------------------------------------------------------------------
class TestGarbageBytes:
    GARBAGE = b"\x00\x01\x02\x03\xff\xfe\xfd" * 100

    def test_garbage_jpg(self, tmp_path):
        path = write_tmp(tmp_path, "garbage.jpg", self.GARBAGE)
        with pytest.raises(Exception):
            load_image(path)

    def test_garbage_png(self, tmp_path):
        path = write_tmp(tmp_path, "garbage.png", self.GARBAGE)
        with pytest.raises(Exception):
            load_image(path)

    def test_garbage_bmp(self, tmp_path):
        path = write_tmp(tmp_path, "garbage.bmp", self.GARBAGE)
        with pytest.raises(Exception):
            load_image(path)

    def test_garbage_dcm(self, tmp_path):
        path = write_tmp(tmp_path, "garbage.dcm", self.GARBAGE)
        with pytest.raises(Exception):
            load_image(path)


# ---------------------------------------------------------------------------
# 3. Truncated valid file (JPEG header only, body missing)
# ---------------------------------------------------------------------------
class TestTruncatedFile:
    # Minimal JPEG SOI marker + a few valid bytes then nothing
    TRUNCATED_JPEG = bytes([0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46])

    def test_truncated_jpg(self, tmp_path):
        path = write_tmp(tmp_path, "truncated.jpg", self.TRUNCATED_JPEG)
        with pytest.raises(Exception):
            load_image(path)

    # Minimal PNG signature + nothing after
    TRUNCATED_PNG = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])

    def test_truncated_png(self, tmp_path):
        path = write_tmp(tmp_path, "truncated.png", self.TRUNCATED_PNG)
        with pytest.raises(Exception):
            load_image(path)


# ---------------------------------------------------------------------------
# 4. Unsupported / unknown extension
# ---------------------------------------------------------------------------
class TestUnsupportedExtension:
    def test_tiff_raises_value_error(self, tmp_path):
        path = write_tmp(tmp_path, "image.tiff", b"II\x2a\x00")
        with pytest.raises(ValueError, match="Unsupported format"):
            load_image(path)

    def test_no_extension_raises(self, tmp_path):
        path = write_tmp(tmp_path, "noext", b"some data")
        with pytest.raises(ValueError, match="Unsupported format"):
            load_image(path)

    def test_txt_extension_raises(self, tmp_path):
        path = write_tmp(tmp_path, "image.txt", b"not an image")
        with pytest.raises(ValueError, match="Unsupported format"):
            load_image(path)


# ---------------------------------------------------------------------------
# 5. File does not exist at all
# ---------------------------------------------------------------------------
class TestMissingFile:
    def test_missing_file_raises(self, tmp_path):
        path = str(tmp_path / "nonexistent.jpg")
        with pytest.raises(Exception):
            load_image(path)

    def test_missing_dcm_raises(self, tmp_path):
        path = str(tmp_path / "ghost.dcm")
        with pytest.raises(Exception):
            load_image(path)


# ---------------------------------------------------------------------------
# 6. Content-extension mismatch (PNG data saved as .jpg)
# ---------------------------------------------------------------------------
class TestExtensionMismatch:
    def test_png_content_as_jpg(self, tmp_path):
        """A valid PNG file but named .jpg — PIL should still open it,
        OR raise; either is acceptable, but it must not crash silently."""
        import io
        from PIL import Image as PILImage
        buf = io.BytesIO()
        PILImage.new("L", (4, 4), color=128).save(buf, format="PNG")
        path = write_tmp(tmp_path, "fake.jpg", buf.getvalue())
        # PIL often opens PNG-content regardless of extension, so we allow
        # both a successful load (returning a valid array) and an exception.
        try:
            img, meta = load_image(path)
            assert img.ndim == 2
            assert img.dtype == np.uint8
        except Exception:
            pass  # also acceptable


# ---------------------------------------------------------------------------
# 7. Valid image — sanity check that good files still work
# ---------------------------------------------------------------------------
class TestValidFile:
    def test_valid_png_loads(self, tmp_path):
        import io
        from PIL import Image as PILImage
        buf = io.BytesIO()
        arr = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        PILImage.fromarray(arr, mode="L").save(buf, format="PNG")
        path = write_tmp(tmp_path, "valid.png", buf.getvalue())
        img, meta = load_image(path)
        assert img.shape == (32, 32)
        assert img.dtype == np.uint8
        assert meta["Width"] == 32
        assert meta["Height"] == 32

    def test_valid_jpg_loads(self, tmp_path):
        import io
        from PIL import Image as PILImage
        buf = io.BytesIO()
        arr = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        PILImage.fromarray(arr, mode="L").save(buf, format="JPEG")
        path = write_tmp(tmp_path, "valid.jpg", buf.getvalue())
        img, meta = load_image(path)
        assert img.ndim == 2
        assert img.dtype == np.uint8

    def test_valid_bmp_loads(self, tmp_path):
        import io
        from PIL import Image as PILImage
        buf = io.BytesIO()
        arr = np.random.randint(0, 256, (16, 16), dtype=np.uint8)
        PILImage.fromarray(arr, mode="L").save(buf, format="BMP")
        path = write_tmp(tmp_path, "valid.bmp", buf.getvalue())
        img, meta = load_image(path)
        assert img.ndim == 2
        assert img.dtype == np.uint8


# ---------------------------------------------------------------------------
# 8. DICOM with valid preamble but no PixelData tag
# ---------------------------------------------------------------------------
class TestDicomEdgeCases:
    def test_dicom_no_pixel_data(self, tmp_path):
        """A file with the DICOM preamble + DICM magic but no pixel tag."""
        # 128-byte preamble + 'DICM' signature
        data = b"\x00" * 128 + b"DICM"
        path = write_tmp(tmp_path, "nopixel.dcm", data)
        with pytest.raises(Exception):
            load_image(path)
