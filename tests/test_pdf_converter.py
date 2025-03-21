"""
tests/test_pdf_converter.py - Test the PDF to image conversion functionality.

This module tests the PDFConverter class from pdf_converter.py, which handles
conversion of PDF receipt files to images suitable for OCR processing.
"""

import os
import sys
import pytest
import numpy as np
from pathlib import Path
import cv2

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the PDFConverter class
from pdf_converter import PDFConverter

# Define paths to test files
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_DIR = os.path.join(BASE_DIR, "Example-Receipts")
SAMPLE_PDF = os.path.join(SAMPLE_DIR, "pdf", "receipt.pdf")
OUTPUT_DIR = os.path.join(BASE_DIR, "test_output")


def setup_module():
    """Set up test environment."""
    # Create output directory for testing
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Ensure the test PDF file exists
    assert os.path.exists(SAMPLE_PDF), f"Sample PDF file not found at {SAMPLE_PDF}"


def teardown_module():
    """Clean up after tests."""
    # Remove output files created during testing
    if os.path.exists(OUTPUT_DIR):
        for filename in os.listdir(OUTPUT_DIR):
            file_path = os.path.join(OUTPUT_DIR, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")


class TestPDFConverter:
    """Tests for the PDFConverter class."""
    
    def setup_method(self):
        """Set up for each test method."""
        self.converter = PDFConverter(
            dpi=300,
            max_pages=5,
            quality=95,
            preferred_engine='auto',  # Use best available
            enhance_mode='auto'
        )
        
        # Create test output directory if needed
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    def test_initialization(self):
        """Test PDFConverter initialization with different parameters."""
        # Test default initialization
        default_converter = PDFConverter()
        assert default_converter.dpi == 300
        assert default_converter.max_pages == 5
        assert default_converter.quality == 100
        assert default_converter.enhance_mode == 'auto'
        
        # Test custom initialization
        custom_converter = PDFConverter(
            dpi=200,
            max_pages=3,
            quality=80,
            preferred_engine='pdf2image',
            enhance_mode='receipt'
        )
        assert custom_converter.dpi == 200
        assert custom_converter.max_pages == 3
        assert custom_converter.quality == 80
        assert custom_converter.enhance_mode == 'receipt'
    
    def test_get_pdf_info(self):
        """Test retrieving PDF information."""
        # Skip if required libraries aren't available
        if not (hasattr(self.converter, 'engine') and 
                (self.converter.engine == 'pypdfium2' or self.converter.engine == 'pdf2image')):
            pytest.skip("No PDF library available")
        
        # Get PDF info
        info = self.converter.get_pdf_info(SAMPLE_PDF)
        
        # Verify basic information
        assert isinstance(info, dict)
        assert 'Pages' in info
        assert info['Pages'] > 0
        assert 'File Size' in info
        assert info['File Size'] > 0
        
        # Print info for debugging
        print("\nPDF Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    def test_convert_pdf_file(self):
        """Test converting a PDF file to images."""
        # Skip if required libraries aren't available
        if not (hasattr(self.converter, 'engine') and 
                (self.converter.engine == 'pypdfium2' or self.converter.engine == 'pdf2image')):
            pytest.skip("No PDF library available")
        
        # Convert without saving
        images = self.converter.convert_pdf_file(
            SAMPLE_PDF,
            enhance=True,
            save_images=False
        )
        
        # Verify result is a list of at least one image
        assert isinstance(images, list)
        assert len(images) > 0
        
        # Check image properties
        first_image = images[0]
        assert isinstance(first_image, np.ndarray)
        assert len(first_image.shape) == 3  # RGB image
        assert first_image.shape[2] == 3    # 3 color channels
        
        # Check image dimensions
        assert first_image.shape[0] > 100  # Height
        assert first_image.shape[1] > 100  # Width
        
        print(f"\nSuccessfully converted PDF with {len(images)} page(s)")
        print(f"First page dimensions: {first_image.shape}")
    
    def test_save_converted_images(self):
        """Test converting and saving PDF images to disk."""
        # Skip if required libraries aren't available
        if not (hasattr(self.converter, 'engine') and 
                (self.converter.engine == 'pypdfium2' or self.converter.engine == 'pdf2image')):
            pytest.skip("No PDF library available")
        
        # Convert and save images
        images = self.converter.convert_pdf_file(
            SAMPLE_PDF,
            output_dir=OUTPUT_DIR,
            output_prefix="test_receipt_",
            enhance=True,
            save_images=True,
            output_format="jpg"
        )
        
        # Verify images were created
        assert len(images) > 0
        
        # Check for files in output directory
        saved_files = [f for f in os.listdir(OUTPUT_DIR) 
                      if f.startswith("test_receipt_") and f.endswith(".jpg")]
        assert len(saved_files) > 0
        assert len(saved_files) == len(images)
        
        # Verify file content
        for file in saved_files:
            file_path = os.path.join(OUTPUT_DIR, file)
            assert os.path.exists(file_path)
            assert os.path.getsize(file_path) > 0
            
            # Try to load the image to verify it's valid
            img = cv2.imread(file_path)
            assert img is not None
            assert img.shape[0] > 0
            assert img.shape[1] > 0
        
        print(f"\nSuccessfully saved {len(saved_files)} images to {OUTPUT_DIR}/")
    
    def test_enhancement_modes(self):
        """Test different enhancement modes."""
        # Skip if required libraries aren't available
        if not (hasattr(self.converter, 'engine') and 
                (self.converter.engine == 'pypdfium2' or self.converter.engine == 'pdf2image')):
            pytest.skip("No PDF library available")
        
        # Test all enhancement modes
        enhancement_modes = ['auto', 'receipt', 'document', 'none']
        
        for mode in enhancement_modes:
            # Convert with specific enhancement mode
            images = self.converter.convert_pdf_file(
                SAMPLE_PDF,
                output_dir=OUTPUT_DIR,
                output_prefix=f"test_{mode}_",
                enhance=True,
                enhance_mode=mode,
                save_images=True,
                output_format="jpg"
            )
            
            assert len(images) > 0
            print(f"\nProcessed PDF with '{mode}' enhancement mode")
            
            # Check for files
            saved_files = [f for f in os.listdir(OUTPUT_DIR) 
                          if f.startswith(f"test_{mode}_") and f.endswith(".jpg")]
            assert len(saved_files) == len(images)
    
    def test_convert_with_page_range(self):
        """Test converting specific page ranges from a PDF."""
        # Skip if required libraries aren't available
        if not (hasattr(self.converter, 'engine') and 
                (self.converter.engine == 'pypdfium2' or self.converter.engine == 'pdf2image')):
            pytest.skip("No PDF library available")
            
        # Get PDF info to check total pages
        info = self.converter.get_pdf_info(SAMPLE_PDF)
        total_pages = info['Pages']
        
        # Skip if PDF has only one page
        if total_pages <= 1:
            pytest.skip("PDF has only one page, cannot test page ranges")
            
        # Test page range (first page only)
        images = self.converter.convert_pdf_file(
            SAMPLE_PDF,
            page_range=(0, 0),  # Only first page (0-indexed)
            enhance=True,
            save_images=False
        )
        
        assert len(images) == 1
        print(f"\nSuccessfully extracted first page with dimensions: {images[0].shape}")
    
    def test_convert_pdf_bytes(self):
        """Test converting PDF from bytes."""
        # Skip if required libraries aren't available
        if not (hasattr(self.converter, 'engine') and 
                (self.converter.engine == 'pypdfium2' or self.converter.engine == 'pdf2image')):
            pytest.skip("No PDF library available")
        
        # Read PDF file as bytes
        with open(SAMPLE_PDF, 'rb') as f:
            pdf_bytes = f.read()
        
        # Convert from bytes
        images = self.converter.convert_pdf_bytes(
            pdf_bytes,
            enhance=True,
            save_images=False
        )
        
        # Verify result
        assert isinstance(images, list)
        assert len(images) > 0
        
        # Check first image
        first_image = images[0]
        assert isinstance(first_image, np.ndarray)
        assert len(first_image.shape) == 3  # RGB image
        assert first_image.shape[2] == 3    # 3 color channels
        
        print(f"\nSuccessfully converted PDF bytes with {len(images)} page(s)")
    
    def test_output_formats(self):
        """Test different output formats (JPG and PNG)."""
        # Skip if required libraries aren't available
        if not (hasattr(self.converter, 'engine') and 
                (self.converter.engine == 'pypdfium2' or self.converter.engine == 'pdf2image')):
            pytest.skip("No PDF library available")
        
        # Test JPG format
        self.converter.convert_pdf_file(
            SAMPLE_PDF,
            output_dir=OUTPUT_DIR,
            output_prefix="test_jpg_",
            save_images=True,
            output_format="jpg"
        )
        
        jpg_files = [f for f in os.listdir(OUTPUT_DIR) 
                    if f.startswith("test_jpg_") and f.endswith(".jpg")]
        assert len(jpg_files) > 0
        
        # Test PNG format
        self.converter.convert_pdf_file(
            SAMPLE_PDF,
            output_dir=OUTPUT_DIR,
            output_prefix="test_png_",
            save_images=True,
            output_format="png"
        )
        
        png_files = [f for f in os.listdir(OUTPUT_DIR) 
                    if f.startswith("test_png_") and f.endswith(".png")]
        assert len(png_files) > 0
        
        print(f"\nSuccessfully tested both JPG and PNG output formats")


if __name__ == "__main__":
    # Run tests when the script is executed directly
    pytest.main(["-xvs", __file__])