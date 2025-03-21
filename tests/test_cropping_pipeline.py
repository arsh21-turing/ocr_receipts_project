#!/usr/bin/env python
"""
test_cropping_pipeline.py - Test script for the cropping pipeline.

This script tests the CroppingPipeline class by processing sample receipt images
from the Example-Receipts/jpg folder and displaying the results.
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse
from typing import List, Dict, Optional

# Add the parent directory to the path so we can import the cropping_pipeline module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the CroppingPipeline class
from cropping_pipeline import CroppingPipeline

# Define path to example receipts
EXAMPLE_RECEIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "receipts_OCR/Example-Receipts")
JPG_DIR = os.path.join(EXAMPLE_RECEIPTS_DIR, "jpg")


def get_sample_images(directory: str = JPG_DIR, count: int = 1) -> List[str]:
    """
    Get sample image paths from the specified directory.
    
    Args:
        directory: Directory containing images.
        count: Number of images to select.
    
    Returns:
        List of image paths.
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    image_paths = []
    for ext in ['*.jpg', '*.jpeg']:
        image_paths.extend(glob.glob(os.path.join(directory, ext)))

    if not image_paths:
        raise FileNotFoundError(f"No JPG images found in {directory}")

    return image_paths[:count]


def display_results(image_path: str, results: Dict[str, np.ndarray], title: str = "Cropping Results") -> None:
    """
    Display the original image and multiple processing results.
    
    Args:
        image_path: Path to the original image.
        results: Dictionary of method names and their resulting images.
        title: Title for the figure.
    """
    n_images = 1 + len(results)
    n_cols = min(3, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols

    plt.figure(figsize=(15, 5 * n_rows))
    plt.suptitle(title, fontsize=16)

    plt.subplot(n_rows, n_cols, 1)
    plt.title("Original")
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.axis('off')

    for i, (method_name, result_img) in enumerate(results.items(), 2):
        plt.subplot(n_rows, n_cols, i)
        plt.title(method_name)
        if result_img is not None:
            result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            plt.imshow(result_rgb)
        plt.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def test_single_image(image_path: str, display: bool = True, save_output: bool = False, output_dir: Optional[str] = None) -> Dict[str, np.ndarray]:
    """
    Test the cropping pipeline on a single image with different methods.
    
    Args:
        image_path: Path to the image.
        display: Whether to display the results.
        save_output: Whether to save output images.
        output_dir: Directory to save output images.
    
    Returns:
        Dictionary of cropping methods and their resulting images.
    """
    print(f"Testing cropping pipeline on: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return {}

    pipeline = CroppingPipeline()

    methods = {
        "Contour Method": "contour",
        "Edge Detection": "edge",
        "Text Block Detection": "text_block",
        "Auto Selection": "auto",
    }

    results = {}

    for method_name, method_code in methods.items():
        try:
            print(f"  Processing with {method_name}...")
            result = pipeline.crop_to_content(img, method=method_code, deskew=True)
            results[method_name] = result

            if save_output:
                if output_dir is None:
                    output_dir = os.path.dirname(image_path)
                os.makedirs(output_dir, exist_ok=True)

                base_name = os.path.splitext(os.path.basename(image_path))[0]
                method_suffix = method_name.lower().replace(" ", "_")
                output_path = os.path.join(output_dir, f"{base_name}_{method_suffix}.jpg")

                cv2.imwrite(output_path, result)
                print(f"  Saved output to: {output_path}")

        except Exception as e:
            print(f"  Error with {method_name}: {e}")
            results[method_name] = None

    if display and results:
        display_results(image_path, results, title=f"Cropping Results - {os.path.basename(image_path)}")

    return results


def test_multiple_images(image_paths: List[str], display: bool = True, save_output: bool = False, output_dir: Optional[str] = None) -> None:
    """
    Test the cropping pipeline on multiple images.
    
    Args:
        image_paths: List of image paths.
        display: Whether to display the results.
        save_output: Whether to save output images.
        output_dir: Directory to save output images.
    """
    print(f"Testing cropping pipeline on {len(image_paths)} images")

    for image_path in image_paths:
        test_single_image(image_path, display, save_output, output_dir)


def main():
    """Main function to run the tests."""
    parser = argparse.ArgumentParser(description="Test the cropping pipeline on receipt images.")
    parser.add_argument("--input", "-i", help="Path to a specific image or directory")
    parser.add_argument("--count", "-c", type=int, default=1, help="Number of random images to test")
    parser.add_argument("--no-display", action="store_true", help="Don't display results")
    parser.add_argument("--save", "-s", action="store_true", help="Save processed images")
    parser.add_argument("--output-dir", "-o", help="Directory to save processed images")
    parser.add_argument("--all", "-a", action="store_true", help="Process all images in the directory")

    args = parser.parse_args()

    if args.input:
        if os.path.isdir(args.input):
            directory = args.input
            count = None if args.all else args.count
            image_paths = get_sample_images(directory, count)
        elif os.path.isfile(args.input):
            image_paths = [args.input]
        else:
            print(f"Error: Input path not found: {args.input}")
            return
    else:
        count = None if args.all else args.count
        image_paths = get_sample_images(JPG_DIR, count)

    test_multiple_images(
        image_paths,
        display=not args.no_display,
        save_output=args.save,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
