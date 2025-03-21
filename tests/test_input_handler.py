"""
test_grayscale_processor.py - Tests for the grayscale_processor module.

This module contains tests for the GrayscaleProcessor class that handles
converting color images to grayscale and applying various enhancements
for improved OCR text extraction.
"""

import os
import sys
import pytest
import numpy as np
import cv2
import glob
from unittest.mock import patch
from receipt_processor.grayscale_processor import GrayscaleProcessor, load_and_process_image

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Define paths to test files
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_DIR = os.path.join(BASE_DIR, "Example-Receipts")
TEST_OUTPUT_DIR = os.path.join(BASE_DIR, "test_output")

# Find all JPG files in the Example-Receipts directory
RECEIPT_IMAGES = glob.glob(os.path.join(SAMPLE_DIR, "**", "*.jpg"), recursive=True)
SAMPLE_JPG = RECEIPT_IMAGES[0] if RECEIPT_IMAGES else None

def setup_module():
    """Set up test environment."""
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

    if not RECEIPT_IMAGES:
        pytest.skip("No receipt images found in Example-Receipts directory")

def teardown_module():
    """Clean up after tests."""
    pass  # Uncomment to remove test outputs: shutil.rmtree(TEST_OUTPUT_DIR)

class TestGrayscaleProcessor:
    """Tests for the GrayscaleProcessor class."""

    def setup_method(self):
        """Set up for each test."""
        self.processor = GrayscaleProcessor()
        
        if RECEIPT_IMAGES:
            self.real_receipt = cv2.imread(RECEIPT_IMAGES[0])
            assert self.real_receipt is not None, f"Failed to load test image: {RECEIPT_IMAGES[0]}"
        else:
            pytest.skip("No receipt images found in Example-Receipts directory")

    def test_to_grayscale_methods(self):
        """Test different grayscale conversion methods."""
        for method in ['weighted', 'average', 'luminosity', 'opencv']:
            gray = self.processor.to_grayscale(self.real_receipt, method=method)
            assert len(gray.shape) == 2
            assert np.min(gray) >= 0 and np.max(gray) <= 255
            cv2.imwrite(os.path.join(TEST_OUTPUT_DIR, f"real_receipt_grayscale_{method}.png"), gray)

    def test_apply_threshold_methods(self):
        """Test different thresholding methods."""
        gray = self.processor.to_grayscale(self.real_receipt)

        for method in ['simple', 'otsu', 'adaptive_mean', 'adaptive_gaussian']:
            binary = self.processor.apply_threshold(gray, method=method)
            assert len(binary.shape) == 2
            cv2.imwrite(os.path.join(TEST_OUTPUT_DIR, f"real_receipt_threshold_{method}.png"), binary)

    def test_denoise(self):
        """Test the denoising functionality."""
        denoised_receipt = self.processor.denoise(self.real_receipt, strength=10)
        cv2.imwrite(os.path.join(TEST_OUTPUT_DIR, "real_receipt_denoised.png"), denoised_receipt)

    def test_sharpen(self):
        """Test the sharpening functionality."""
        gray = self.processor.to_grayscale(self.real_receipt)
        sharpened_receipt = self.processor.sharpen(gray, amount=1.0)
        cv2.imwrite(os.path.join(TEST_OUTPUT_DIR, "real_receipt_sharpened.png"), sharpened_receipt)

    def test_enhance_for_ocr(self):
        """Test the full OCR enhancement pipeline."""
        enhanced_receipt = self.processor.enhance_for_ocr(
            self.real_receipt,
            denoise_strength=10,
            sharpen_amount=0.5,
            threshold_method='adaptive_gaussian'
        )
        cv2.imwrite(os.path.join(TEST_OUTPUT_DIR, "real_receipt_enhanced.png"), enhanced_receipt)

    def test_process_for_receipt_ocr(self):
        """Test receipt-specific processing."""
        processed_receipt = self.processor.process_for_receipt_ocr(self.real_receipt)
        cv2.imwrite(os.path.join(TEST_OUTPUT_DIR, "real_receipt_processed.png"), processed_receipt)

    def test_process_for_document_ocr(self):
        """Test document-specific processing."""
        processed_receipt = self.processor.process_for_document_ocr(self.real_receipt)
        cv2.imwrite(os.path.join(TEST_OUTPUT_DIR, "real_receipt_document_processed.png"), processed_receipt)

class TestRealReceiptProcessing:
    """Tests focused specifically on processing real receipt images."""

    @pytest.mark.skipif(len(RECEIPT_IMAGES) == 0, reason="No receipt images found")
    def test_multiple_real_receipts(self):
        """Test processing multiple real receipt images."""
        processor = GrayscaleProcessor()

        for i, receipt_path in enumerate(RECEIPT_IMAGES[:5]):  
            receipt = cv2.imread(receipt_path)
            if receipt is None:
                continue

            filename = os.path.basename(receipt_path)
            base_name = os.path.splitext(filename)[0]

            methods = {
                'original': receipt,
                'grayscale': processor.to_grayscale(receipt),
                'receipt_ocr': processor.process_for_receipt_ocr(receipt),
                'document_ocr': processor.process_for_document_ocr(receipt),
                'enhanced': processor.enhance_for_ocr(receipt)
            }

            for method_name, processed_img in methods.items():
                cv2.imwrite(os.path.join(TEST_OUTPUT_DIR, f"{base_name}_{method_name}.png"), processed_img)

class TestUtilityFunctions:
    """Tests for utility functions in the grayscale_processor module."""

    @pytest.mark.skipif(len(RECEIPT_IMAGES) == 0, reason="No receipt images found")
    def test_load_and_process_image_real_file(self):
        """Test the load_and_process_image function with a real file."""
        processed = load_and_process_image(RECEIPT_IMAGES[0], enhancement_type='auto')

        filename = os.path.basename(RECEIPT_IMAGES[0])
        base_name = os.path.splitext(filename)[0]
        cv2.imwrite(os.path.join(TEST_OUTPUT_DIR, f"{base_name}_auto_processed.png"), processed)

    def test_invalid_enhancement_type(self):
        """Test handling of invalid enhancement type."""
        with pytest.raises(ValueError, match="Invalid enhancement type"):
            load_and_process_image(RECEIPT_IMAGES[0], enhancement_type='invalid')

if __name__ == "__main__":
    pytest.main(["-v", __file__])
