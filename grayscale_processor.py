import cv2
import numpy as np
from typing import Optional, Tuple, Union
import logging

# Set up logging
logger = logging.getLogger(__name__)

class GrayscaleProcessor:
    """
    Handles conversion of color images to grayscale and applies various
    enhancement techniques to improve OCR accuracy.
    
    This class provides methods to:
    - Convert images to grayscale using different algorithms
    - Apply preprocessing for text enhancement
    - Apply adaptive thresholding for better text extraction
    - Perform noise reduction and image sharpening
    
    Attributes:
        default_method (str): Default grayscale conversion method
        default_threshold_method (str): Default thresholding method
    """
    
    def __init__(self, 
                 default_method: str = 'weighted',
                 default_threshold_method: str = 'adaptive_gaussian'):
        """
        Initialize a GrayscaleProcessor instance.
        
        Args:
            default_method: Default grayscale conversion method
                ('weighted', 'average', 'luminosity', or 'opencv')
            default_threshold_method: Default thresholding method
                ('simple', 'otsu', 'adaptive_mean', 'adaptive_gaussian')
        """
        self.default_method = default_method
        self.default_threshold_method = default_threshold_method
        
        # Validate the default methods
        valid_gray_methods = ['weighted', 'average', 'luminosity', 'opencv']
        valid_thresh_methods = ['simple', 'otsu', 'adaptive_mean', 'adaptive_gaussian']
        
        if default_method not in valid_gray_methods:
            raise ValueError(f"Invalid grayscale method: {default_method}. "
                           f"Valid methods are: {', '.join(valid_gray_methods)}")
        
        if default_threshold_method not in valid_thresh_methods:
            raise ValueError(f"Invalid threshold method: {default_threshold_method}. "
                           f"Valid methods are: {', '.join(valid_thresh_methods)}")
    
    def to_grayscale(self, image: np.ndarray, method: Optional[str] = None) -> np.ndarray:
        """
        Convert a color image to grayscale using the specified method.
        
        Args:
            image: Input color image (RGB or BGR)
            method: Grayscale conversion method 
                ('weighted', 'average', 'luminosity', 'opencv', or None to use default)
                
        Returns:
            Grayscale image as numpy array
        
        Raises:
            ValueError: If the image is not a color image or method is invalid
        """
        if len(image.shape) < 3 or image.shape[2] < 3:
            # Already grayscale, return as is
            logger.info("Image is already grayscale, returning as is")
            return image
        
        method = method or self.default_method
        
        if method == 'opencv':
            # Use OpenCV's built-in conversion
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        elif method == 'average':
            # Simple average of RGB channels
            return np.mean(image, axis=2).astype(np.uint8)
        
        elif method == 'luminosity':
            # Weight channels based on human perception (ITU-R BT.601)
            # Y = 0.299 R + 0.587 G + 0.114 B
            return np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        
        elif method == 'weighted':
            # Weighted average optimized for text visibility
            # More weight to green channel which typically shows better text contrast
            return np.dot(image[..., :3], [0.25, 0.65, 0.1]).astype(np.uint8)
        
        else:
            raise ValueError(f"Invalid grayscale method: {method}")
    
    def apply_threshold(self, 
                       image: np.ndarray, 
                       method: Optional[str] = None,
                       block_size: int = 11,
                       c: int = 2,
                       threshold_value: int = 127) -> np.ndarray:
        """
        Apply thresholding to a grayscale image to improve text visibility.
        
        Args:
            image: Grayscale image
            method: Thresholding method 
                ('simple', 'otsu', 'adaptive_mean', 'adaptive_gaussian', or None to use default)
            block_size: Size of pixel neighborhood for adaptive thresholding (must be odd)
            c: Constant subtracted from the mean/weighted mean in adaptive thresholding
            threshold_value: Threshold value for simple thresholding (0-255)
                
        Returns:
            Thresholded binary image
            
        Raises:
            ValueError: If the image is not grayscale or method is invalid
        """
        # Ensure the image is grayscale
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = self.to_grayscale(image)
        
        # Ensure block_size is odd
        if block_size % 2 == 0:
            block_size += 1
        
        method = method or self.default_threshold_method
        
        if method == 'simple':
            # Simple binary thresholding
            _, binary = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
            return binary
        
        elif method == 'otsu':
            # Otsu's method automatically determines optimal threshold
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary
        
        elif method == 'adaptive_mean':
            # Adaptive thresholding using mean of neighborhood
            return cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                cv2.THRESH_BINARY, block_size, c
            )
        
        elif method == 'adaptive_gaussian':
            # Adaptive thresholding using Gaussian-weighted mean
            return cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, block_size, c
            )
        
        else:
            raise ValueError(f"Invalid threshold method: {method}")
    
    def denoise(self, image: np.ndarray, strength: int = 10) -> np.ndarray:
        """
        Apply noise reduction to improve OCR accuracy.
        
        Args:
            image: Input image (grayscale or color)
            strength: Denoising strength parameter
                
        Returns:
            Denoised image
        """
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Color image
            return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
        else:
            # Grayscale image
            return cv2.fastNlMeansDenoising(image, None, strength, 7, 21)
    
    def sharpen(self, image: np.ndarray, kernel_size: int = 3, amount: float = 1.0) -> np.ndarray:
        """
        Sharpen an image to enhance text edges.
        
        Args:
            image: Input image
            kernel_size: Size of the sharpening kernel
            amount: Strength of the sharpening effect
                
        Returns:
            Sharpened image
        """
        # Create a sharpening kernel
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure kernel size is odd
            
        kernel = np.zeros((kernel_size, kernel_size), np.float32)
        kernel[kernel_size//2, kernel_size//2] = 2.0
        kernel = kernel - np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        kernel = kernel * amount
        
        # Apply the kernel
        sharpened = cv2.filter2D(image, -1, kernel)
        
        # Clip values to valid range
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    def enhance_for_ocr(self, 
                       image: np.ndarray, 
                       denoise_strength: int = 10,
                       sharpen_amount: float = 0.5,
                       threshold_method: Optional[str] = None) -> np.ndarray:
        """
        Apply a full enhancement pipeline optimized for OCR.
        
        Args:
            image: Input color or grayscale image
            denoise_strength: Strength of denoising
            sharpen_amount: Amount of sharpening
            threshold_method: Method for thresholding
                
        Returns:
            Enhanced image optimized for OCR
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = self.to_grayscale(image)
        else:
            gray = image.copy()
        
        # Apply denoising
        denoised = self.denoise(gray, denoise_strength)
        
        # Apply sharpening
        sharpened = self.sharpen(denoised, amount=sharpen_amount)
        
        # Apply thresholding
        binary = self.apply_threshold(sharpened, method=threshold_method)
        
        return binary
    
    def deskew(self, image: np.ndarray, max_angle: float = 45.0) -> np.ndarray:
        """
        Attempts to correct skewed (rotated) text in the image.
        
        Args:
            image: Input image (grayscale or binary)
            max_angle: Maximum angle to correct in degrees
                
        Returns:
            Deskewed image
        """
        # Ensure image is grayscale
        if len(image.shape) == 3:
            gray = self.to_grayscale(image)
        else:
            gray = image.copy()
        
        # Threshold the image
        if np.max(gray) > 1:  # Check if the image is already binary
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        else:
            thresh = gray
        
        # Calculate skew angle
        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]
        
        # Adjust angle
        if angle < -max_angle:
            angle = -(90 + angle)
        elif angle > max_angle:
            angle = 90 - angle
        else:
            angle = -angle
        
        # Rotate the image to correct the skew
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), 
                                 flags=cv2.INTER_CUBIC, 
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=255)
        
        return rotated
    
    def process_for_receipt_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Special processing pipeline specifically for receipt OCR.
        
        Args:
            image: Input receipt image (color or grayscale)
                
        Returns:
            Processed image optimized for receipt OCR
        """
        # Convert to grayscale
        gray = self.to_grayscale(image, method='weighted')
        
        # Apply contrast enhancement using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Denoise with low strength to preserve text details
        denoised = self.denoise(enhanced, strength=5)
        
        # Apply adaptive thresholding
        binary = self.apply_threshold(
            denoised, 
            method='adaptive_gaussian',
            block_size=15,
            c=8
        )
        
        # Try to deskew if needed
        deskewed = self.deskew(binary)
        
        return deskewed
    
    def process_for_document_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Special processing pipeline for document OCR (non-receipt documents).
        
        Args:
            image: Input document image (color or grayscale)
                
        Returns:
            Processed image optimized for document OCR
        """
        # Convert to grayscale using luminosity method for better text contrast
        gray = self.to_grayscale(image, method='luminosity')
        
        # Apply mild denoising
        denoised = self.denoise(gray, strength=7)
        
        # Sharpen to enhance text edges
        sharpened = self.sharpen(denoised, amount=0.3)
        
        # Apply Otsu thresholding which works well for documents
        binary = self.apply_threshold(sharpened, method='otsu')
        
        # Attempt to deskew if needed
        deskewed = self.deskew(binary)
        
        return deskewed


def load_and_process_image(image_path: str, 
                          enhancement_type: str = 'auto',
                          deskew: bool = True) -> np.ndarray:
    """
    Utility function to load and process an image in one step.
    
    Args:
        image_path: Path to the input image
        enhancement_type: Type of enhancement to apply 
            ('auto', 'receipt', 'document', 'none')
        deskew: Whether to attempt to correct image skew
    
    Returns:
        Processed image ready for OCR
    """
    # Create processor
    processor = GrayscaleProcessor()
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")
    
    # Process based on enhancement type
    if enhancement_type == 'auto':
        # Try to automatically detect if it's a receipt or a document
        # This is a simplistic approach - real detection would be more sophisticated
        # Receipts tend to be narrow and tall
        h, w = image.shape[:2]
        if h > w * 1.8:  # If height is significantly greater than width
            return processor.process_for_receipt_ocr(image)
        else:
            return processor.process_for_document_ocr(image)
    
    elif enhancement_type == 'receipt':
        return processor.process_for_receipt_ocr(image)
    
    elif enhancement_type == 'document':
        return processor.process_for_document_ocr(image)
    
    elif enhancement_type == 'none':
        # Just convert to grayscale without enhancements
        return processor.to_grayscale(image)
    
    else:
        raise ValueError(f"Invalid enhancement type: {enhancement_type}")