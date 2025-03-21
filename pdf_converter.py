"""
pdf_converter.py - PDF to image conversion for restaurant receipt OCR processor.

This module provides specialized functionality for converting PDF receipts
to high-quality images suitable for OCR processing. It supports multiple
PDF conversion libraries with automatic fallback and includes various
enhancement options to improve OCR accuracy.
"""

import os
import tempfile
import uuid
import logging
from typing import List, Dict, Union, Optional, Tuple, Any
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# Configure PDF converter libraries with fallback options
try:
    import pypdfium2 as pdfium
    PDFIUM_AVAILABLE = True
except ImportError:
    PDFIUM_AVAILABLE = False
    
try:
    from pdf2image import convert_from_path, convert_from_bytes, pdfinfo_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PDFConverter:
    """
    Handles conversion of PDF files to images optimized for OCR processing.
    
    This class provides methods to:
    - Convert PDF files or bytes to high-quality images
    - Enhance images specifically for receipt OCR
    - Apply different preprocessing techniques based on receipt characteristics
    - Support multiple PDF conversion libraries with automatic fallback
    
    Attributes:
        dpi (int): Resolution (DPI) for PDF rendering
        max_pages (int): Maximum number of PDF pages to process
        quality (int): JPEG quality (0-100) for image output
        preferred_engine (str): Preferred PDF engine ('auto', 'pypdfium2', 'pdf2image')
        tmp_dir (str): Temporary directory for file operations
        enhance_mode (str): Default enhancement mode ('auto', 'receipt', 'document', 'none')
    """
    
    def __init__(self, 
                 dpi: int = 300,
                 max_pages: int = 5,
                 quality: int = 100,
                 preferred_engine: str = 'auto',
                 tmp_dir: Optional[str] = None,
                 enhance_mode: str = 'auto'):
        """
        Initialize a PDFConverter instance.
        
        Args:
            dpi: Resolution in DPI for PDF rendering (higher = better quality but larger files)
            max_pages: Maximum number of PDF pages to process (limit for large documents)
            quality: JPEG quality (0-100) for PDF page images
            preferred_engine: PDF conversion engine preference ('auto', 'pypdfium2', 'pdf2image')
            tmp_dir: Temporary directory for file operations
            enhance_mode: Default enhancement mode ('auto', 'receipt', 'document', 'none')
        """
        self.dpi = dpi
        self.max_pages = max_pages
        self.quality = quality
        self.tmp_dir = tmp_dir or tempfile.gettempdir()
        self.enhance_mode = enhance_mode
        
        # Set PDF conversion engine
        if preferred_engine == 'auto':
            # Use pypdfium2 if available (faster and more robust)
            self.engine = 'pypdfium2' if PDFIUM_AVAILABLE else 'pdf2image'
        else:
            self.engine = preferred_engine
        
        # Validate that we have at least one PDF conversion library available
        if not (PDFIUM_AVAILABLE or PDF2IMAGE_AVAILABLE):
            logger.warning(
                "No PDF conversion libraries available. "
                "Install pypdfium2 or pdf2image to enable PDF conversion."
            )
    
    def convert_pdf_file(self, 
                         pdf_path: str,
                         output_dir: Optional[str] = None, 
                         output_prefix: str = "page_", 
                         page_range: Optional[Tuple[int, int]] = None,
                         enhance: bool = True,
                         enhance_mode: Optional[str] = None,
                         save_images: bool = False,
                         output_format: str = "jpg") -> List[np.ndarray]:
        """
        Convert a PDF file to a list of images.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save images if save_images is True
            output_prefix: Prefix for saved image filenames
            page_range: Tuple of (start_page, end_page) to process (0-indexed)
            enhance: Whether to enhance images for better OCR
            enhance_mode: Mode for image enhancement (overrides default)
            save_images: Whether to save the images to disk
            output_format: Format for saved images ('jpg', 'png')
            
        Returns:
            List of numpy arrays containing the images (RGB format)
            
        Raises:
            FileNotFoundError: If PDF file not found
            ValueError: If PDF conversion fails
        """
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Use class default for enhance_mode if not specified
        enhance_mode = enhance_mode or self.enhance_mode
        
        # Convert PDF to images using selected engine
        if self.engine == 'pypdfium2' and PDFIUM_AVAILABLE:
            images = self._convert_with_pdfium(
                pdf_path, 
                page_range=page_range
            )
        elif PDF2IMAGE_AVAILABLE:
            images = self._convert_with_pdf2image(
                pdf_path, 
                page_range=page_range
            )
        else:
            raise ValueError(
                "Cannot convert PDF: No PDF conversion libraries available. "
                "Please install pypdfium2 or pdf2image."
            )
        
        # Apply enhancement if requested
        if enhance and enhance_mode != 'none':
            images = [self.enhance_image(img, mode=enhance_mode) for img in images]
        
        # Save images if requested
        if save_images:
            self.save_images(images, output_dir, output_prefix, output_format)
            
        return images
    
    def convert_pdf_bytes(self, 
                          pdf_bytes: bytes,
                          output_dir: Optional[str] = None,
                          output_prefix: str = "page_",
                          page_range: Optional[Tuple[int, int]] = None,
                          enhance: bool = True,
                          enhance_mode: Optional[str] = None,
                          save_images: bool = False,
                          output_format: str = "jpg") -> List[np.ndarray]:
        """
        Convert PDF bytes to a list of images.
        
        Args:
            pdf_bytes: PDF file content as bytes
            output_dir: Directory to save images if save_images is True
            output_prefix: Prefix for saved image filenames
            page_range: Tuple of (start_page, end_page) to process (0-indexed)
            enhance: Whether to enhance images for better OCR
            enhance_mode: Mode for image enhancement (overrides default)
            save_images: Whether to save the images to disk
            output_format: Format for saved images ('jpg', 'png')
            
        Returns:
            List of numpy arrays containing the images (RGB format)
            
        Raises:
            ValueError: If PDF conversion fails
        """
        if not pdf_bytes:
            raise ValueError("Empty PDF bytes provided")
        
        # Use class default for enhance_mode if not specified
        enhance_mode = enhance_mode or self.enhance_mode
        
        # Create a temporary file if using pypdfium or we have to fall back
        if (self.engine == 'pypdfium2' and PDFIUM_AVAILABLE) or \
           (self.engine == 'auto' and PDFIUM_AVAILABLE) or \
           (not PDF2IMAGE_AVAILABLE):
            # For pypdfium, we need to save to a file first
            temp_path = os.path.join(self.tmp_dir, f"temp_pdf_{uuid.uuid4()}.pdf")
            try:
                with open(temp_path, 'wb') as f:
                    f.write(pdf_bytes)
                images = self.convert_pdf_file(
                    temp_path,
                    output_dir=output_dir,
                    output_prefix=output_prefix,
                    page_range=page_range,
                    enhance=enhance,
                    enhance_mode=enhance_mode,
                    save_images=save_images,
                    output_format=output_format
                )
                return images
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        
        # Direct conversion using pdf2image if available
        elif PDF2IMAGE_AVAILABLE:
            # Convert PDF bytes to PIL Images using pdf2image
            try:
                # Determine page range for pdf2image
                first_page = page_range[0] + 1 if page_range else 1
                last_page = page_range[1] + 1 if page_range else self.max_pages
                
                pil_images = convert_from_bytes(
                    pdf_bytes,
                    dpi=self.dpi,
                    first_page=first_page,
                    last_page=last_page,
                    thread_count=2,  # Limit threads for web server use
                    use_cropbox=True,
                    fmt='jpeg',
                    jpegopt={'quality': self.quality, 'optimize': True}
                )
                
                # Convert PIL Images to numpy arrays
                images = [np.array(img) for img in pil_images]
                
                # Apply enhancement if requested
                if enhance and enhance_mode != 'none':
                    images = [self.enhance_image(img, mode=enhance_mode) for img in images]
                
                # Save images if requested
                if save_images:
                    self.save_images(images, output_dir, output_prefix, output_format)
                    
                return images
            
            except Exception as e:
                logger.error(f"PDF2Image conversion error: {e}")
                raise ValueError(f"Failed to convert PDF bytes: {e}")
        
        else:
            raise ValueError(
                "Cannot convert PDF: No PDF conversion libraries available. "
                "Please install pypdfium2 or pdf2image."
            )
    
    def enhance_image(self, 
                      image: np.ndarray, 
                      mode: str = 'auto') -> np.ndarray:
        """
        Enhance an image to improve OCR quality.
        
        This method applies various preprocessing techniques to make text more readable,
        particularly for receipt images that may have low contrast or noise.
        
        Args:
            image: Image as numpy array (RGB format)
            mode: Enhancement mode:
                - 'auto': Automatically detect best enhancement based on image characteristics
                - 'receipt': Optimize for thermal receipt printer output (high contrast)
                - 'document': Optimize for standard document scans
                - 'none': No enhancement
                
        Returns:
            Enhanced image as numpy array (RGB format)
        """
        if mode == 'none':
            return image
        
        # Convert to grayscale for processing
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
            
        # Auto-detect the best enhancement method if 'auto' is specified
        if mode == 'auto':
            # Calculate image statistics to determine the best approach
            mean_val = np.mean(gray)
            std_val = np.std(gray)
            
            # Classify the image type based on statistics
            if std_val < 50:  # Low contrast image (likely a receipt)
                mode = 'receipt'
            else:
                mode = 'document'
        
        # Apply receipt-specific enhancements
        if mode == 'receipt':
            return self._enhance_receipt(image, gray)
            
        # Apply document-specific enhancements
        elif mode == 'document':
            return self._enhance_document(image, gray)
        
        # Default case: return original image
        return image
    
    def _enhance_receipt(self, 
                         image: np.ndarray, 
                         gray: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply receipt-specific enhancements.
        
        Optimized for thermal printer receipts with potentially low contrast
        or faded text. Uses adaptive thresholding which works well for receipts.
        
        Args:
            image: Original image in RGB format
            gray: Optional pre-computed grayscale image
            
        Returns:
            Enhanced image in RGB format
        """
        # Convert to grayscale if not provided
        if gray is None:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 1. Apply adaptive thresholding (works well for receipts)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 21, 10
        )
        
        # 2. Remove noise using morphological operations
        kernel = np.ones((1, 1), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 3. For very faded receipts, try CLAHE first
        if np.std(gray) < 30:  # Very low contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced_gray = clahe.apply(gray)
            # Then apply adaptive threshold
            binary = cv2.adaptiveThreshold(
                enhanced_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 21, 10
            )
            
        # 4. Convert back to RGB for consistency
        enhanced = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        return enhanced
    
    def _enhance_document(self, 
                          image: np.ndarray, 
                          gray: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply document-specific enhancements.
        
        Optimized for standard document scans, focusing on preserving
        text clarity while reducing noise.
        
        Args:
            image: Original image in RGB format
            gray: Optional pre-computed grayscale image
            
        Returns:
            Enhanced image in RGB format
        """
        # Convert to grayscale if not provided
        if gray is None:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
        # 1. Apply bilateral filtering (preserves edges while reducing noise)
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # 2. Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(filtered)
        
        # 3. Sharpen the image to make text clearer
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced_gray, -1, kernel)
        
        # 4. Convert back to RGB
        enhanced = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)
        return enhanced
    
    def _convert_with_pdfium(self, 
                             pdf_path: str,
                             page_range: Optional[Tuple[int, int]] = None) -> List[np.ndarray]:
        """
        Convert PDF to images using pypdfium2.
        
        Args:
            pdf_path: Path to the PDF file
            page_range: Tuple of (start_page, end_page) to process (0-indexed)
            
        Returns:
            List of numpy arrays, one per page
            
        Raises:
            ValueError: If PDF conversion fails
        """
        try:
            # Open the PDF file
            pdf = pdfium.PdfDocument(pdf_path)
            
            # Get page range to process
            num_pages = len(pdf)
            if num_pages == 0:
                raise ValueError("PDF has no pages")
                
            # Determine start and end page (0-indexed)
            start_page = 0
            end_page = min(num_pages, self.max_pages) - 1
            
            if page_range:
                start_page = max(0, min(page_range[0], num_pages - 1))
                end_page = min(page_range[1], num_pages - 1, start_page + self.max_pages - 1)
            
            # Calculate scale factor based on DPI (72 DPI is the base for PDF)
            scale = self.dpi / 72.0
            
            # Convert each page
            images = []
            for page_idx in range(start_page, end_page + 1):
                page = pdf[page_idx]
                width = int(page.get_width() * scale)
                height = int(page.get_height() * scale)
                
                # Render the page to a bitmap
                bitmap = page.render(
                    width=width,
                    height=height,
                    scale=scale,
                )
                
                # Convert to PIL Image and then to numpy array
                pil_image = bitmap.to_pil()
                np_image = np.array(pil_image)
                
                images.append(np_image)
                
            return images
                
        except Exception as e:
            logger.error(f"PyPDFium2 conversion error: {e}")
            # Fall back to pdf2image if available
            if PDF2IMAGE_AVAILABLE:
                logger.info("Falling back to pdf2image for PDF conversion")
                return self._convert_with_pdf2image(pdf_path, page_range)
            raise ValueError(f"Failed to convert PDF with pypdfium2: {e}")
    
    def _convert_with_pdf2image(self, 
                               pdf_path: str,
                               page_range: Optional[Tuple[int, int]] = None) -> List[np.ndarray]:
        """
        Convert PDF to images using pdf2image (poppler-based).
        
        Args:
            pdf_path: Path to the PDF file
            page_range: Tuple of (start_page, end_page) to process (0-indexed)
            
        Returns:
            List of numpy arrays, one per page
            
        Raises:
            ValueError: If PDF conversion fails
        """
        try:
            # Try to get info about the PDF
            try:
                pdf_info = pdfinfo_from_path(pdf_path)
                num_pages = pdf_info['Pages']
            except:
                # If pdfinfo fails, just try with max pages
                num_pages = self.max_pages
            
            # Determine start and end page (1-indexed for pdf2image)
            start_page = 1
            end_page = min(num_pages, self.max_pages)
            
            if page_range:
                # Convert from 0-indexed to 1-indexed
                start_page = max(1, page_range[0] + 1)
                end_page = min(num_pages, page_range[1] + 1, 
                              start_page + self.max_pages - 1)
                
            # Convert PDF to list of PIL Images
            pil_images = convert_from_path(
                pdf_path, 
                dpi=self.dpi, 
                first_page=start_page,
                last_page=end_page,
                thread_count=2,  # Limit threads for web server use
                use_cropbox=True,
                grayscale=False,
                fmt='jpeg',
                jpegopt={'quality': self.quality, 'optimize': True}
            )
            
            # Convert PIL Images to numpy arrays
            images = [np.array(img) for img in pil_images]
            return images
            
        except Exception as e:
            logger.error(f"PDF2Image conversion error: {e}")
            raise ValueError(f"Failed to convert PDF with pdf2image: {e}")
    
    def save_images(self, 
                   images: List[np.ndarray], 
                   output_dir: Optional[str] = None,
                   output_prefix: str = "page_",
                   output_format: str = "jpg") -> List[str]:
        """
        Save a list of images to disk.
        
        Args:
            images: List of numpy arrays (RGB format)
            output_dir: Directory to save images (default: current directory)
            output_prefix: Prefix for image filenames
            output_format: Format to save images ('jpg' or 'png')
            
        Returns:
            List of paths to saved images
            
        Raises:
            ValueError: If output format is not supported
        """
        if not images:
            return []
            
        # Validate output format
        if output_format.lower() not in ['jpg', 'jpeg', 'png']:
            raise ValueError(f"Unsupported output format: {output_format}. Use 'jpg' or 'png'")
            
        # Use current directory if output_dir not specified
        output_dir = output_dir or os.getcwd()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Normalize file extension
        ext = output_format.lower()
        if ext == 'jpeg':
            ext = 'jpg'
            
        # Save each image
        saved_paths = []
        for i, img in enumerate(images):
            # Convert from RGB to BGR for OpenCV
            bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Generate filename
            filename = f"{output_prefix}{i:03d}.{ext}"
            file_path = os.path.join(output_dir, filename)
            
            # Save image
            if ext == 'jpg':
                cv2.imwrite(file_path, bgr_img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            else:  # png
                cv2.imwrite(file_path, bgr_img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                
            saved_paths.append(file_path)
            
        return saved_paths
    
    def get_pdf_info(self, pdf_path: str) -> Dict[str, Any]:
        """
        Get information about a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing PDF metadata
            
        Raises:
            FileNotFoundError: If PDF file is not found
            ValueError: If PDF info cannot be extracted
        """
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        try:
            # Try with pdf2image first if available
            if PDF2IMAGE_AVAILABLE:
                try:
                    pdf_info = pdfinfo_from_path(pdf_path)
                    return {
                        'Pages': pdf_info['Pages'],
                        'Title': pdf_info.get('Title', ''),
                        'Author': pdf_info.get('Author', ''),
                        'Creator': pdf_info.get('Creator', ''),
                        'Producer': pdf_info.get('Producer', ''),
                        'Format': pdf_info.get('Format', ''),
                        'Encrypted': pdf_info.get('Encrypted', False),
                        'Page Size': pdf_info.get('Page size', ''),
                        'File Size': os.path.getsize(pdf_path)
                    }
                except Exception as e:
                    logger.warning(f"Failed to get PDF info with pdf2image: {e}")
                    # Fall back to pypdfium
                    pass
                    
            # Try with pypdfium if available
            if PDFIUM_AVAILABLE:
                try:
                    pdf = pdfium.PdfDocument(pdf_path)
                    page_count = len(pdf)
                    
                    # For first page dimensions
                    page_size = ""
                    if page_count > 0:
                        page = pdf[0]
                        page_size = f"{page.get_width()} x {page.get_height()} pts"
                    
                    return {
                        'Pages': page_count,
                        'File Size': os.path.getsize(pdf_path),
                        'Page Size': page_size
                    }
                except Exception as e:
                    logger.error(f"Failed to get PDF info with pypdfium: {e}")
            
            # If we get here, both methods failed
            raise ValueError("Failed to extract PDF information. Check if the file is a valid PDF.")
            
        except Exception as e:
            logger.error(f"Error getting PDF info: {e}")
            raise ValueError(f"Failed to get PDF info: {e}")


# Example usage
if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert PDF to images.')
    parser.add_argument('pdf_path', help='Path to PDF file')
    parser.add_argument('--output_dir', '-o', default='output', help='Output directory')
    parser.add_argument('--dpi', '-d', type=int, default=300, help='DPI for PDF rendering')
    parser.add_argument('--enhance', '-e', choices=['auto', 'receipt', 'document', 'none'], 
                        default='auto', help='Enhancement mode')
    parser.add_argument('--format', '-f', choices=['jpg', 'png'], default='jpg', 
                       help='Output image format')
    parser.add_argument('--start_page', '-s', type=int, default=0, 
                       help='Start page (0-indexed)')
    parser.add_argument('--end_page', '-p', type=int, default=None, 
                       help='End page (0-indexed, inclusive)')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.pdf_path):
        print(f"Error: File not found: {args.pdf_path}")
        sys.exit(1)
    
    # Create PDF converter
    converter = PDFConverter(
        dpi=args.dpi,
        enhance_mode=args.enhance
    )
    
    # Set page range
    page_range = (args.start_page, args.end_page) if args.end_page is not None else None
    
    try:
        # Get PDF info
        info = converter.get_pdf_info(args.pdf_path)
        print(f"PDF Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Convert PDF
        print(f"Converting PDF to {args.format.upper()} images...")
        images = converter.convert_pdf_file(
            args.pdf_path,
            output_dir=args.output_dir,
            page_range=page_range,
            enhance=True,
            enhance_mode=args.enhance,
            save_images=True,
            output_format=args.format
        )
        
        print(f"Successfully converted {len(images)} pages to {args.output_dir}/")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)