"""
cropping_pipeline.py - Module for detecting and cropping relevant sections from receipt and document images.

This module provides functions to:

- Detect region of interest in images
- Remove unnecessary margins and backgrounds
- Extract content areas based on different cropping strategies
- Provide a pipeline for automatic content extraction
"""

import cv2
import numpy as np
import math
import os
import glob
from typing import List, Tuple, Optional, Dict, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('cropping_pipeline')

# Type definitions
BoundingBox = Tuple[int, int, int, int]  # (x, y, width, height)
Point = Tuple[int, int]
Contour = np.ndarray

class CroppingPipeline:
    """
    Pipeline for detecting and cropping relevant content from images.
    """

    def __init__(self, min_content_area_ratio: float = 0.01, padding: int = 10):
        """
        Initialize the cropping pipeline.
        
        Args:
            min_content_area_ratio: Minimum ratio of contour area to image area to be considered content
            padding: Number of pixels to add around detected content when cropping
        """
        self.min_content_area_ratio = min_content_area_ratio
        self.padding = padding

    def crop_to_content(self, image: np.ndarray, method: str = 'auto', deskew: bool = True) -> np.ndarray:
        """
        Detect and crop to relevant content in the image.
        
        Args:
            image: Input image (BGR format)
            method: Cropping method ('auto', 'contour', 'edge', 'text_block')
            deskew: Whether to deskew the image before cropping
        
        Returns:
            Cropped image
        """
        img = image.copy()
        h, w = img.shape[:2]
        img_area = h * w

        if deskew:
            img = self.deskew_image(img)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

        if method == 'auto':
            method = 'contour' if h > 1.5 * w else 'edge'
        
        if method == 'contour':
            bbox = self._crop_by_contour(gray, img_area)
        elif method == 'edge':
            bbox = self._crop_by_edge_detection(gray)
        elif method == 'text_block':
            bbox = self._crop_to_text_blocks(gray, img_area)
        else:
            raise ValueError(f"Unknown cropping method: {method}")

        if bbox is None:
            logger.warning("No content area detected, returning original image.")
            return img
        
        x, y, width, height = bbox
        x_start, y_start = max(0, x - self.padding), max(0, y - self.padding)
        x_end, y_end = min(w, x + width + self.padding), min(h, y + height + self.padding)
        
        return img[y_start:y_end, x_start:x_end]

    def _crop_by_contour(self, gray_img: np.ndarray, img_area: int) -> Optional[BoundingBox]:
        """
        Detect content area using contour detection.
        
        Args:
            gray_img: Grayscale input image
            img_area: Total area of the image
        
        Returns:
            Bounding box of detected content area or None if no content detected
        """
        _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > img_area * self.min_content_area_ratio]
        if not filtered_contours:
            return None
        
        all_points = np.concatenate(filtered_contours)
        return cv2.boundingRect(all_points)

    def _crop_by_edge_detection(self, gray_img: np.ndarray) -> Optional[BoundingBox]:
        """
        Detect content area using edge detection.
        
        Args:
            gray_img: Grayscale input image
        
        Returns:
            Bounding box of detected content area or None if no content detected
        """
        blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        all_points = np.concatenate(contours)
        return cv2.boundingRect(all_points)

    def deskew_image(self, image: np.ndarray) -> np.ndarray:
        """
        Deskew an image to straighten text.
        
        Args:
            image: Input image
        
        Returns:
            Deskewed image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        lines = cv2.HoughLinesP(binary, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
        
        if lines is None or len(lines) < 5:
            return image
        
        angles = [math.atan2(y2 - y1, x2 - x1) * 180 / np.pi for x1, y1, x2, y2 in lines[:, 0] if x2 - x1 != 0]
        angles = [angle for angle in angles if abs(angle) < 45]
        if not angles:
            return image
        
        median_angle = np.median(angles)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    def detect(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Wrapper for compatibility. Returns bounding box only.
        """
        cropped = self.crop_to_content(image, method='auto', deskew=True)
        if cropped.shape == image.shape:
            return None  # No cropping was applied
        else:
            # Calculate bounding box difference
            h_img, w_img = image.shape[:2]
            h_crop, w_crop = cropped.shape[:2]

            # Find top-left offset by comparing pixels
            y_offset = next((i for i in range(h_img) if not np.array_equal(cropped[0], image[i])), 0)
            x_offset = next((j for j in range(w_img) if not np.array_equal(cropped[:, 0], image[:, j])), 0)

            return (x_offset, y_offset, w_crop, h_crop)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Crop images to content.")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("--output", "-o", help="Output image path")
    parser.add_argument("--method", "-m", choices=["auto", "contour", "edge", "text_block"], default="auto")
    args = parser.parse_args()
    
    img = cv2.imread(args.input)
    pipeline = CroppingPipeline()
    cropped = pipeline.crop_to_content(img, method=args.method)
    if args.output:
        cv2.imwrite(args.output, cropped)
    print(f"Image cropped successfully. Shape: {cropped.shape[:2]}")
