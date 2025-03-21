import streamlit as st
import cv2
import numpy as np
import os
import glob
import sys
import tempfile
import random
from PIL import Image

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from grayscale_processor import GrayscaleProcessor, load_and_process_image
from pdf_converter import PDFConverter  # Use our existing PDFConverter
from cropping_pipeline import CroppingPipeline  # Import the cropping pipeline

# Define Example Receipts Path (Root Directory)
EXAMPLE_RECEIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "Example-Receipts"))

def get_example_receipt_paths():
    """Retrieve example receipt image paths from structured subfolders."""
    if not os.path.exists(EXAMPLE_RECEIPTS_DIR):
        st.error(f"❌ Example-Receipts directory not found: {EXAMPLE_RECEIPTS_DIR}")
        return {}

    receipt_types = {}
    for root, _, files in os.walk(EXAMPLE_RECEIPTS_DIR):
        category = os.path.relpath(root, EXAMPLE_RECEIPTS_DIR)
        for file in files:
            if file.lower().endswith(("jpg", "jpeg", "png", "bmp")):
                # Fix typo in filenames
                if "reciept" in file.lower():  
                    corrected_name = file.lower().replace("reciept", "receipt")
                    os.rename(os.path.join(root, file), os.path.join(root, corrected_name))
                    file = corrected_name 

                if category not in receipt_types:
                    receipt_types[category] = []
                receipt_types[category].append(os.path.join(root, file))

    if not receipt_types:
        st.error("⚠ No images found in Example-Receipts subfolders.")

    return receipt_types

def process_image(image, enhancement_type, grayscale_method, processor, progress_bar=None, filename=None):
    """Process image with enhancement options and display results."""
    if progress_bar is None:
        progress_bar = st.progress(0)
    
    if enhancement_type != "none":
        st.subheader(f"Enhanced Image (Type: {enhancement_type.capitalize()})")
        progress_bar.progress(60)
        
        if enhancement_type == "auto":
            processed = load_and_process_image(
                image_path=None, 
                enhancement_type=enhancement_type,
                deskew=True,
                image=image  
            )
        elif enhancement_type == "receipt":
            processed = processor.process_for_receipt_ocr(image)
        elif enhancement_type == "document":
            processed = processor.process_for_document_ocr(image)
        
        progress_bar.progress(80)
        st.image(processed, use_container_width=True)
        
        result = cv2.imencode('.png', processed)[1].tobytes()
        st.download_button(
            label="Download Enhanced Image",
            data=result,
            file_name=f"enhanced_{enhancement_type}_{os.path.splitext(filename)[0]}.png" if filename else "enhanced_image.png",
            mime="image/png"
        )
        
    else:
        st.subheader(f"Grayscale Image (Method: {grayscale_method})")
        progress_bar.progress(60)
        
        gray = processor.to_grayscale(image, method=grayscale_method)
        st.image(gray, use_container_width=True)
        
        progress_bar.progress(80)
        
        final_result = cv2.imencode('.png', gray)[1].tobytes()
        st.download_button(
            label="Download Processed Image",
            data=final_result,
            file_name=f"processed_{os.path.splitext(filename)[0]}.png" if filename else "processed_image.png",
            mime="image/png"
        )

def crop_image(image, cropping_method, apply_deskew, pipeline, filename=None):
    """Crop the image using the selected method and display results."""
    progress_bar = st.progress(30)
    
    try:
        # Process according to selected method
        result = pipeline.crop_to_content(
            image, 
            method=cropping_method,
            deskew=apply_deskew
        )
        
        progress_bar.progress(70)
        st.subheader(f"Cropped Image (Method: {cropping_method})")
        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        # Calculate reduction percentage
        h, w = image.shape[:2]
        h_result, w_result = result.shape[:2]
        reduction = (1 - (h_result * w_result) / (h * w)) * 100
        
        # Show image details
        st.write(f"**Original size:** {w} × {h} pixels")
        st.write(f"**Cropped size:** {w_result} × {h_result} pixels")
        st.write(f"**Area reduction:** {reduction:.1f}%")
        
        # Download button
        final_result = cv2.imencode('.png', result)[1].tobytes()
        st.download_button(
            label="Download Cropped Image",
            data=final_result,
            file_name=f"cropped_{cropping_method}_{os.path.splitext(filename)[0]}.png" if filename else "cropped_image.png",
            mime="image/png"
        )
        
        progress_bar.progress(100)
        return result
        
    except Exception as e:
        st.error(f"Error cropping image: {str(e)}")
        progress_bar.progress(100)
        return image

def main():
    st.set_page_config(
        page_title="OCR Image Processor",
        page_icon="📝",
        layout="wide"
    )
    
    st.title("📝 OCR Image Processor")
    st.write("Upload an image or PDF and apply various processing techniques to enhance it for OCR.")
    
    # Check if Example-Receipts directory exists
    example_receipts = get_example_receipt_paths()
    has_examples = len(example_receipts) > 0

    if has_examples:
        total_files = sum(len(v) for v in example_receipts.values())
        st.success(f"✅ Found {total_files} example receipts in the Example-Receipts directory.")

    # Initialize processors
    processor = GrayscaleProcessor()
    pdf_converter = PDFConverter()
    cropping_pipeline = CroppingPipeline()  # Initialize the cropping pipeline

    # Sidebar for processing options
    st.sidebar.header("Processing Options")
    
    # Add a radio button to select between enhancement and cropping
    processing_mode = st.sidebar.radio("Processing Mode", ["Enhancement", "Cropping"])
    
    # Conditional display of options based on the selected mode
    if processing_mode == "Enhancement":
        grayscale_method = st.sidebar.selectbox("Grayscale Conversion Method", 
                                             ["weighted", "luminosity", "average", "opencv"], 
                                             index=0)
        enhancement_type = st.sidebar.selectbox("Enhancement Type", 
                                             ["none", "receipt", "document", "auto"], 
                                             index=0)
    else:  # Cropping mode
        cropping_method = st.sidebar.selectbox("Cropping Method", 
                                            ["auto", "contour", "edge", "text_block"], 
                                            index=0,
                                            help="Method used to detect content boundaries")
        apply_deskew = st.sidebar.checkbox("Apply Deskew", value=True,
                                        help="Straighten the image before cropping")
        padding = st.sidebar.slider("Padding (pixels)", 0, 50, 10,
                                  help="Extra space to add around detected content")
        # Set padding in the pipeline
        cropping_pipeline.padding = padding

    # File upload section
    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader("Choose an image or PDF file", type=["jpg", "jpeg", "png", "bmp", "pdf"])

    with col2:
        if has_examples:
            st.write("### Test with Example Receipt")
            receipt_type = st.selectbox("Receipt Type", options=list(example_receipts.keys()), help="Select a category of example receipts to test")
            test_mode = st.radio("Selection Mode", ["Random Receipt", "Select Specific Receipt"])

            if test_mode == "Select Specific Receipt":
                file_options = [os.path.basename(path) for path in example_receipts[receipt_type]]
                selected_file = st.selectbox("Select Receipt File", file_options)
                specific_path = next((path for path in example_receipts[receipt_type] if os.path.basename(path) == selected_file), None)
                test_specific = st.button("🧪 Test with Selected Receipt")
            else:
                test_specific = False
                specific_path = None
            
            test_random = st.button("🧪 Test with Random Receipt" if test_mode == "Random Receipt" else "")

            test_btn = test_random or test_specific
        else:
            st.warning("No example receipts found. Add images to the 'Example-Receipts' directory to enable testing.")
            test_btn = False
            receipt_type = None
            test_specific = False
            specific_path = None

    # Process uploaded file
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    pdf_path = tmp_file.name
                
                # Convert PDF to image
                images = pdf_converter.convert_pdf_file(
                    pdf_path,
                    page_range=(0, 0),  # Just first page for now
                    save_images=False
                )
                
                if not images:
                    st.error("No pages extracted from PDF.")
                    return
                
                image = images[0]
                os.unlink(pdf_path)  # Clean up temp file
            except Exception as e:
                st.error(f"Error processing PDF: {e}")
                return
        else:
            # Process regular image file
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if image is None:
                st.error("Could not read the image file.")
                return

        st.subheader("Original Image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        # Process according to selected mode
        if processing_mode == "Enhancement":
            process_image(image, enhancement_type, grayscale_method, processor, filename=uploaded_file.name)
        else:  # Cropping mode
            crop_image(image, cropping_method, apply_deskew, cropping_pipeline, filename=uploaded_file.name)

    # Process example receipt if test button is clicked
    elif test_btn and has_examples and receipt_type:
        if test_specific and specific_path:
            example_path = specific_path
        else:
            example_path = random.choice(example_receipts[receipt_type])

        st.write("### Testing with Example Receipt")
        st.write(f"**Selected file:** {os.path.relpath(example_path, EXAMPLE_RECEIPTS_DIR)}")

        image = cv2.imread(example_path)
        if image is None:
            st.error(f"Could not read the example file: {example_path}")
            return

        st.subheader("Original Example Receipt")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        # Process according to selected mode
        if processing_mode == "Enhancement":
            process_image(image, enhancement_type, grayscale_method, processor, filename=os.path.basename(example_path))
        else:  # Cropping mode
            crop_image(image, cropping_method, apply_deskew, cropping_pipeline, filename=os.path.basename(example_path))
    
    # Add information about the cropping methods when in cropping mode
    if processing_mode == "Cropping" and not uploaded_file and not test_btn:
        st.info("""
        ### About Image Cropping
        
        The cropping feature automatically detects and extracts the relevant content from your images, 
        removing unnecessary backgrounds and empty spaces.
        
        **Available methods:**
        - **Auto**: Automatically selects the best method based on image content
        - **Contour**: Best for receipts with clear boundaries
        - **Edge**: Good for documents with distinct edges
        - **Text Block**: Focuses on preserving text areas
        
        **Benefits:**
        - Improved OCR accuracy
        - Reduced file size
        - Better visual presentation
        """)

if __name__ == "__main__":
    main()