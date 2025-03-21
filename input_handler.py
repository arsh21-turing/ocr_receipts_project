"""
Implementation of the input handler for the restaurant receipt OCR processor.

This module contains the InputHandler class that handles loading
and processing of receipt files in various formats (JPEG, PNG, PDF).
It includes functionality to:
1. Validate file types and sizes
2. Load images from files or bytes
3. Convert PDFs to images with enhanced quality control
4. Save uploaded files
"""

import os
import tempfile
import uuid
import logging
from pathlib import Path
from typing import List, Tuple, Union, Optional, BinaryIO

import cv2
import numpy as np
from PIL import Image

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


class InputHandler:
    """
    Handles input file loading and validation for receipt processing.
    
    This class provides methods to:
    - Validate file extensions and sizes
    - Load image files (JPEG, PNG)
    - Convert PDF files to images
    - Process uploaded files
    
    Attributes:
        max_file_size_bytes (int): Maximum allowed file size in bytes
        pdf_dpi (int): DPI to use when converting PDF to images
        max_pdf_pages (int): Maximum number of PDF pages to process
        tmp_dir (str): Temporary directory for file operations
        allowed_extensions (list): List of allowed file extensions
    """
    
    def __init__(self, 
                 max_file_size_mb: int = 10, 
                 pdf_dpi: int = 300, 
                 max_pdf_pages: int = 5,
                 pdf_quality: int = 100,
                 preferred_pdf_engine: str = 'auto',
                 tmp_dir: Optional[str] = None):
        """
        Initialize an InputHandler instance.
        
        Args:
            max_file_size_mb: Maximum allowed file size in MB
            pdf_dpi: DPI to use when converting PDF to images
            max_pdf_pages: Maximum number of PDF pages to process
            pdf_quality: JPEG quality (0-100) to use when converting PDF pages
            preferred_pdf_engine: PDF conversion engine ('auto', 'pypdfium2', or 'pdf2image')
            tmp_dir: Temporary directory for file operations
        """
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.pdf_dpi = pdf_dpi
        self.max_pdf_pages = max_pdf_pages
        self.pdf_quality = pdf_quality
        self.tmp_dir = tmp_dir or tempfile.gettempdir()
        self.allowed_extensions = ['.jpg', '.jpeg', '.png', '.pdf']
        
        # Set PDF conversion engine
        if preferred_pdf_engine == 'auto':
            # Use pypdfium2 if available (faster and more robust)
            self.pdf_engine = 'pypdfium2' if PDFIUM_AVAILABLE else 'pdf2image'
        else:
            self.pdf_engine = preferred_pdf_engine
        
        # Validate that we have at least one PDF conversion method available
        if not (PDFIUM_AVAILABLE or PDF2IMAGE_AVAILABLE):
            logger.warning(
                "No PDF conversion libraries available. "
                "Install pypdfium2 or pdf2image to enable PDF support."
            )
    
    def validate_file_extension(self, filename: str) -> Tuple[bool, str]:
        """
        Validate that the file has an allowed extension.
        
        Args:
            filename: The name of the file to validate
            
        Returns:
            Tuple containing (is_valid, message)
        """
        ext = os.path.splitext(filename)[1].lower()
        if ext in self.allowed_extensions:
            return True, f"Valid file extension: {ext}"
        else:
            return False, f"Invalid file extension: {ext}. Allowed: {', '.join(self.allowed_extensions)}"
    
    def validate_file_size(self, file_size: int) -> bool:
        """
        Validate that the file size is within allowed limits.
        
        Args:
            file_size: Size of the file in bytes
            
        Returns:
            True if file size is valid, False otherwise
        """
        return file_size <= self.max_file_size_bytes
    
    def load(self, file_path: str) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Load a file from the specified path.
        
        Args:
            file_path: Path to the file to load
            
        Returns:
            For images (JPEG, PNG): numpy array of the image
            For PDFs: list of numpy arrays, one per page
            
        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file has an invalid extension or format
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        valid, message = self.validate_file_extension(file_path)
        if not valid:
            raise ValueError(message)
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if not self.validate_file_size(file_size):
            max_size_mb = self.max_file_size_bytes / (1024 * 1024)
            raise ValueError(f"File too large: {file_size / (1024 * 1024):.2f}MB. Maximum allowed: {max_size_mb}MB")
        
        # Load file based on extension
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext in ['.jpg', '.jpeg', '.png']:
            # Load image file
            return self._load_image_file(file_path)
        elif ext == '.pdf':
            # Convert PDF to images
            return self._convert_pdf_to_images(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
    
    def load_bytes(self, file_bytes: bytes, filename: str) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Load a file from bytes.
        
        Args:
            file_bytes: Bytes of the file
            filename: Name of the file (used to determine type)
            
        Returns:
            For images (JPEG, PNG): numpy array of the image
            For PDFs: list of numpy arrays, one per page
            
        Raises:
            ValueError: If the file has an invalid extension or format
        """
        valid, message = self.validate_file_extension(filename)
        if not valid:
            raise ValueError(message)
        
        # Check file size
        file_size = len(file_bytes)
        if not self.validate_file_size(file_size):
            max_size_mb = self.max_file_size_bytes / (1024 * 1024)
            raise ValueError(f"File too large: {file_size / (1024 * 1024):.2f}MB. Maximum allowed: {max_size_mb}MB")
        
        # Load file based on extension
        ext = os.path.splitext(filename)[1].lower()
        
        if ext in ['.jpg', '.jpeg', '.png']:
            # Load image from bytes
            return self._load_image_bytes(file_bytes)
        elif ext == '.pdf':
            # Convert PDF bytes to images
            return self._convert_pdf_bytes_to_images(file_bytes)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
    
    def save_upload(self, file_obj: BinaryIO, filename: str, upload_dir: str) -> str:
        """
        Save an uploaded file to the specified directory.
        
        Args:
            file_obj: File object to save
            filename: Name of the file
            upload_dir: Directory to save the file to
            
        Returns:
            Path to the saved file
        """
        # Validate file extension
        valid, message = self.validate_file_extension(filename)
        if not valid:
            raise ValueError(message)
        
        # Create the upload directory if it doesn't exist
        os.makedirs(upload_dir, exist_ok=True)
        
        # Generate a unique filename to avoid overwrites
        ext = os.path.splitext(filename)[1].lower()
        unique_filename = f"{str(uuid.uuid4())}{ext}"
        file_path = os.path.join(upload_dir, unique_filename)
        
        # Save the file
        with open(file_path, 'wb') as f:
            f.write(file_obj.read())
        
        # Check the file size
        file_size = os.path.getsize(file_path)
        if not self.validate_file_size(file_size):
            # If the file is too large, delete it and raise an error
            os.unlink(file_path)
            max_size_mb = self.max_file_size_bytes / (1024 * 1024)
            raise ValueError(f"File too large: {file_size / (1024 * 1024):.2f}MB. Maximum allowed: {max_size_mb}MB")
        
        return file_path
    
    def _load_image_file(self, file_path: str) -> np.ndarray:
        """
        Load an image file using OpenCV.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            numpy array containing the image in RGB format
        """
        # Read the image using OpenCV
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError(f"Failed to load image: {file_path}")
        
        # Convert from BGR to RGB (OpenCV loads as BGR by default)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def _load_image_bytes(self, image_bytes: bytes) -> np.ndarray:
        """
        Load an image from bytes using OpenCV.
        
        Args:
            image_bytes: Bytes of the image
            
        Returns:
            numpy array containing the image in RGB format
        """
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image bytes")
        
        # Convert from BGR to RGB
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    def convert_pdf(self, pdf_path_or_bytes: Union[str, bytes], enhance: bool = True,
                    enhance_method: str = 'auto') -> List[np.ndarray]:
        """
        Public method to convert PDF to images with optional enhancement.
        
        Args:
            pdf_path_or_bytes: Path to PDF file or PDF bytes
            enhance: Whether to enhance the images for better OCR
            enhance_method: Method to use for enhancement ('auto', 'receipt', 'document', 'none')
            
        Returns:
            List of numpy arrays, one per page
            
        Raises:
            ValueError: If PDF conversion fails
            FileNotFoundError: If PDF file is not found
        """
        # Handle input type
        if isinstance(pdf_path_or_bytes, str):
            # It's a file path
            images = self._convert_pdf_to_images(pdf_path_or_bytes)
        elif isinstance(pdf_path_or_bytes, bytes):
            # It's PDF bytes
            images = self._convert_pdf_bytes_to_images(pdf_path_or_bytes)
        else:
            raise TypeError("pdf_path_or_bytes must be a string path or bytes")
        
        # Skip enhancement if not requested or if we already enhanced in the converters
        if not enhance or enhance_method == 'none':
            return images
            
        # Otherwise manually enhance each image
        enhanced_images = []
        for img in images:
            enhanced_images.append(self.enhance_pdf_image(img, enhance_method=enhance_method))
            
        return enhanced_images
    
    def enhance_pdf_image(self, image: np.ndarray, enhance_method: str = 'auto') -> np.ndarray:
        """
        Enhance a PDF-extracted image to improve OCR quality.
        
        This method applies various preprocessing to make text more readable in PDF scans,
        particularly for receipt images that may have low contrast or noise.
        
        Args:
            image: Image as numpy array (RGB)
            enhance_method: Method to use ('auto', 'receipt', 'document', 'none')
            
        Returns:
            Enhanced image as numpy array (RGB)
        """
        if enhance_method == 'none':
            return image
            
        # Convert to grayscale for processing
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
            
        # Auto-detect the best enhancement method if 'auto' is specified
        if enhance_method == 'auto':
            # Calculate image statistics to determine the best approach
            mean_val = np.mean(gray)
            std_val = np.std(gray)
            
            # Classify the image type based on statistics
            if std_val < 50:  # Low contrast image
                enhance_method = 'receipt'
            else:
                enhance_method = 'document'
                
        # Select enhancement method
        if enhance_method == 'receipt':
            # Receipt-specific enhancements (thermal printer output)
            # 1. Apply adaptive thresholding (works well for receipts)
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 21, 10
            )
            
            # 2. Denoise using morphological operations
            kernel = np.ones((1, 1), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # 3. Convert back to RGB
            enhanced = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
            return enhanced
            
        elif enhance_method == 'document':
            # Document-specific enhancements (better for color documents)
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
        
        # Default case: return original image
        return image
    
    def _convert_pdf_to_images(self, pdf_path: str) -> List[np.ndarray]:
        """
        Convert a PDF file to a list of images using the selected conversion engine.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of numpy arrays, one per page
            
        Raises:
            ValueError: If no PDF conversion libraries are available or if PDF conversion fails
        """
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        if self.pdf_engine == 'pypdfium2' and PDFIUM_AVAILABLE:
            return self._convert_pdf_pdfium(pdf_path)
        elif PDF2IMAGE_AVAILABLE:
            return self._convert_pdf_pdf2image(pdf_path)
        else:
            raise ValueError(
                "Cannot convert PDF: No PDF conversion libraries available. "
                "Please install pypdfium2 or pdf2image."
            )
    
    def _convert_pdf_bytes_to_images(self, pdf_bytes: bytes) -> List[np.ndarray]:
        """
        Convert PDF bytes to a list of images using the selected conversion engine.
        
        Args:
            pdf_bytes: Bytes of the PDF file
            
        Returns:
            List of numpy arrays, one per page
            
        Raises:
            ValueError: If no PDF conversion libraries are available or if PDF conversion fails
        """
        if not pdf_bytes:
            raise ValueError("Empty PDF bytes provided")
            
        # Create a temporary file for the PDF bytes if needed
        if (self.pdf_engine == 'pypdfium2' and PDFIUM_AVAILABLE) or \
           (self.pdf_engine != 'pdf2image' and not PDF2IMAGE_AVAILABLE):
            # For pypdfium, we need to save to a file first
            temp_path = os.path.join(self.tmp_dir, f"temp_pdf_{uuid.uuid4()}.pdf")
            try:
                with open(temp_path, 'wb') as f:
                    f.write(pdf_bytes)
                return self._convert_pdf_to_images(temp_path)
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        elif PDF2IMAGE_AVAILABLE:
            # Use pdf2image direct bytes conversion
            try:
                pil_images = convert_from_bytes(
                    pdf_bytes,
                    dpi=self.pdf_dpi,
                    first_page=1,
                    last_page=self.max_pdf_pages,
                    thread_count=2,  # Limit threads for web server use
                    use_cropbox=True,
                    fmt='jpeg',
                    jpegopt={'quality': self.pdf_quality, 'optimize': True}
                )
                
                # Convert and enhance each image
                images = []
                for img in pil_images:
                    # Convert to numpy array
                    np_img = np.array(img)
                    # Enhance image for better OCR
                    enhanced_img = self.enhance_pdf_image(np_img, enhance_method='auto')
                    images.append(enhanced_img)
                    
                return images
            except Exception as e:
                logger.error(f"PDF2Image conversion error: {e}")
                raise ValueError(f"Failed to convert PDF bytes: {e}")
        else:
            raise ValueError(
                "Cannot convert PDF: No PDF conversion libraries available. "
                "Please install pypdfium2 or pdf2image."
            )
            
    def _convert_pdf_pdfium(self, pdf_path: str) -> List[np.ndarray]:
        """
        Convert a PDF file to images using pypdfium2.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of numpy arrays, one per page
        """
        try:
            # Open the PDF file
            pdf = pdfium.PdfDocument(pdf_path)
            
            # Limit to max pages
            page_count = min(len(pdf), self.max_pdf_pages)
            if page_count == 0:
                raise ValueError("PDF has no pages")
                
            # Calculate scale factor based on DPI (72 DPI is the base for PDF)
            scale = self.pdf_dpi / 72.0
            
            # Convert each page
            images = []
            for page_idx in range(page_count):
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
                
                # Enhance the image for better OCR results
                enhanced_image = self.enhance_pdf_image(np_image, enhance_method='auto')
                images.append(enhanced_image)
                
            return images
                
        except Exception as e:
            logger.error(f"PyPDFium2 conversion error: {e}")
            # Fall back to pdf2image if available
            if PDF2IMAGE_AVAILABLE:
                logger.info("Falling back to pdf2image for PDF conversion")
                return self._convert_pdf_pdf2image(pdf_path)
            raise ValueError(f"Failed to convert PDF with pypdfium2: {e}")
    
    def _convert_pdf_pdf2image(self, pdf_path: str) -> List[np.ndarray]:
        """
        Convert a PDF file to images using pdf2image (poppler-based).
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of numpy arrays, one per page
        """
        try:
            # Try to get info about the PDF
            try:
                pdf_info = pdfinfo_from_path(pdf_path)
                page_count = min(pdf_info['Pages'], self.max_pdf_pages)
            except:
                # If pdfinfo fails, just try with max pages
                page_count = self.max_pdf_pages
                
            # Convert PDF to list of PIL Images
            pil_images = convert_from_path(
                pdf_path, 
                dpi=self.pdf_dpi, 
                first_page=1,
                last_page=page_count,
                thread_count=2,  # Limit threads for web server use
                use_cropbox=True,
                grayscale=False,
                fmt='jpeg',
                jpegopt={'quality': self.pdf_quality, 'optimize': True}
            )
            
            # Convert PIL Images to numpy arrays and enhance them
            images = []
            for img in pil_images:
                # Convert to numpy array
                np_img = np.array(img)
                # Enhance image for better OCR
                enhanced_img = self.enhance_pdf_image(np_img, enhance_method='auto')
                images.append(enhanced_img)
                
            return images
            
        except Exception as e:
            logger.error(f"PDF2Image conversion error: {e}")
            raise ValueError(f"Failed to convert PDF with pdf2image: {e}")


if __name__ == "__main__":
    # Simple test code
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python input_handler.py <file_path>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    handler = InputHandler()
    
    try:
        print(f"Processing {file_path}...")
        result = handler.load(file_path)
        
        if isinstance(result, list):
            print(f"Successfully loaded PDF with {len(result)} pages")
            # Save the first page as a test
            if len(result) > 0:
                output_path = "output_page_0.jpg"
                cv2.imwrite(output_path, cv2.cvtColor(result[0], cv2.COLOR_RGB2BGR))
                print(f"Saved first page to {output_path}")
        else:
            print(f"Successfully loaded image of shape {result.shape}")
            # Save the image as a test
            output_path = "output_image.jpg"
            cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            print(f"Saved image to {output_path}")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)