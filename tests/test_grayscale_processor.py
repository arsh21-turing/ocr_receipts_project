import sys
import os
import pytest
import numpy as np
import cv2
import glob
from unittest.mock import patch

# ✅ Add the **project root directory (receipts_OCR)** to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ✅ Import the GrayscaleProcessor class **after updating sys.path**
from receipt_processor.grayscale_processor import GrayscaleProcessor, load_and_process_image

# ✅ Define base directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
EXAMPLE_DIR = os.path.join(BASE_DIR, "Example-Receipts")
TEST_OUTPUT_DIR = os.path.join(BASE_DIR, "test_output")

# ✅ Find available JPG files in Example-Receipts
RECEIPT_IMAGES = glob.glob(os.path.join(EXAMPLE_DIR, "**", "*.jpg"), recursive=True)
SAMPLE_JPG = RECEIPT_IMAGES[0] if RECEIPT_IMAGES else None

def setup_module():
    """Set up test environment."""
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    if not RECEIPT_IMAGES:
        pytest.skip("No receipt images found in Example-Receipts directory")

def teardown_module():
    """Clean up after tests."""
    pass  # Uncomment to remove test output after running

class TestGrayscaleProcessor:
    """Tests for the GrayscaleProcessor class."""

    def setup_method(self):
        """Set up for each test."""
        self.processor = GrayscaleProcessor()
        if SAMPLE_JPG:
            self.real_receipt = cv2.imread(SAMPLE_JPG)
        else:
            self.real_receipt = None

    def test_to_grayscale_methods(self):
        """Test different grayscale conversion methods."""
        if self.real_receipt is None:
            pytest.skip("Skipping test: No real receipt images available")
        for method in ['weighted', 'average', 'luminosity', 'opencv']:
            gray = self.processor.to_grayscale(self.real_receipt, method=method)
            assert len(gray.shape) == 2
            assert 0 <= np.min(gray) <= np.max(gray) <= 255
            cv2.imwrite(os.path.join(TEST_OUTPUT_DIR, f"real_receipt_grayscale_{method}.jpg"), gray)

if __name__ == "__main__":
    pytest.main(["-v", __file__])
