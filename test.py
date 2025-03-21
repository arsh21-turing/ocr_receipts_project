# final_app.py with LLM integration (improved security)
import streamlit as st
import cv2
import numpy as np
import os
import sys
import io
from pdf2image import convert_from_bytes
from PIL import Image
import json
import requests

# Import your cropping and preprocessing modules
from cropping_pipeline import CroppingPipeline

# Try importing grayscale preprocessing
try:
    from grayscale_processor import preprocess_image
except ImportError:
    try:
        from grayscale_processor import GrayscaleProcessor
        grayscale_processor = GrayscaleProcessor()

        def preprocess_image(image, adaptive=True):
            if hasattr(grayscale_processor, 'preprocess'):
                return grayscale_processor.preprocess(image, adaptive=adaptive)
            elif hasattr(grayscale_processor, 'process'):
                return grayscale_processor.process(image, adaptive=adaptive)
            elif hasattr(grayscale_processor, 'process_image'):
                return grayscale_processor.process_image(image, adaptive=adaptive)
            else:
                if adaptive:
                    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                else:
                    _, processed = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    return processed
    except ImportError:
        def preprocess_image(image, adaptive=True):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            if adaptive:
                return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            else:
                _, processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                return processed

# PaddleOCR
from paddleocr import PaddleOCR
ocr_engine = PaddleOCR(use_angle_cls=True, lang='en')

st.set_page_config(page_title="Receipt OCR App", page_icon="🧾", layout="wide")

@st.cache_resource
def get_cropping_pipeline():
    return CroppingPipeline()

def pil_to_cv2(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_image):
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

def get_receipt_detection_method(cropping_pipeline):
    for attr in ['detect_receipt', 'detect', 'detect_crop', 'find_receipt', 'get_crop_coordinates', 'crop']:
        if hasattr(cropping_pipeline, attr):
            return getattr(cropping_pipeline, attr)
    st.warning("CroppingPipeline doesn't have a recognized detection method. Using fallback implementation.")
    return lambda img: None

def run_ocr(image):
    result = ocr_engine.ocr(image, cls=True)
    extracted_text = "\n".join([line[1][0] for line in result[0]]) if result else ""
    return result, extracted_text

def process_image_with_cropping(image, crop_enabled=True, preprocess_enabled=True, preprocess_mode="Standard"):
    cropping_pipeline = get_cropping_pipeline()
    detect_method = get_receipt_detection_method(cropping_pipeline)
    results = {"original": image.copy()}

    # Preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    if preprocess_enabled:
        if preprocess_mode == "Standard":
            preprocessed = preprocess_image(gray, adaptive=True)
        elif preprocess_mode == "Enhanced":
            preprocessed = cv2.equalizeHist(preprocess_image(gray, adaptive=True))
        else:
            gray = cv2.bilateralFilter(gray, 11, 17, 17)
            preprocessed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        working_image = preprocessed
        results["preprocessed"] = preprocessed
    else:
        working_image = gray
        results["preprocessed"] = gray

    if crop_enabled:
        try:
            crop_coords = detect_method(working_image)
            if crop_coords is None:
                results["cropped"] = image
                results["cropped_preprocessed"] = working_image
            elif isinstance(crop_coords, (tuple, list)) and len(crop_coords) == 4:
                x, y, w, h = crop_coords
                results["cropped"] = image[y:y+h, x:x+w]
                results["cropped_preprocessed"] = working_image[y:y+h, x:x+w]
                bbox = image.copy()
                cv2.rectangle(bbox, (x, y), (x+w, y+h), (0, 255, 0), 2)
                results["bbox"] = bbox
            elif isinstance(crop_coords, np.ndarray):
                x, y, w, h = cv2.boundingRect(crop_coords)
                results["cropped"] = image[y:y+h, x:x+w]
                results["cropped_preprocessed"] = working_image[y:y+h, x:x+w]
                bbox = image.copy()
                cv2.rectangle(bbox, (x, y), (x+w, y+h), (0, 255, 0), 2)
                results["bbox"] = bbox
            else:
                results["cropped"] = image
                results["cropped_preprocessed"] = working_image
        except Exception as e:
            st.error(f"Cropping error: {e}")
            results["cropped"] = image
            results["cropped_preprocessed"] = working_image
    else:
        results["cropped"] = image
        results["cropped_preprocessed"] = working_image

    return results

def process_with_llm(text, api_key, model="gpt-3.5-turbo"):
    """Process OCR text with OpenAI API to extract structured information."""
    if not api_key or not text:
        return None
    
    try:
        # Direct API call using requests, not setting a global API key
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        prompt = f"""
        Extract structured data from this receipt text. 
        If it's a receipt, extract:
        - Store/Merchant name
        - Date
        - Total amount
        - Individual items with prices if available
        - Tax amount if available
        - Payment method if available
        
        If it's an invoice, extract:
        - Company name
        - Invoice number
        - Date
        - Due date
        - Total amount
        - Line items with descriptions and prices
        - Tax amount
        - Payment terms
        
        Format the response as a JSON object.
        
        Text from receipt:
        {text}
        """
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that extracts structured data from receipt and invoice images."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        # Check if the request was successful
        if response.status_code == 200:
            response_data = response.json()
            result = response_data['choices'][0]['message']['content']
            
            # Try to parse as JSON, but fall back to text if not valid JSON
            try:
                json_result = json.loads(result)
                return json_result
            except json.JSONDecodeError:
                # Return the text as is if it's not valid JSON
                return {"raw_response": result}
        else:
            return {"error": f"API Error: {response.status_code}", "details": response.text}
            
    except Exception as e:
        return {"error": str(e)}

# Sidebar
with st.sidebar:
    st.header("Settings")
    lang_options = ["en", "ch", "fr", "german", "korean", "japanese"]
    ocr_lang = st.selectbox("OCR Language", lang_options, index=0)
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    enable_cropping = st.checkbox("Enable Auto-Cropping", value=True)
    do_preprocess = st.checkbox("Enable Preprocessing", value=True)
    if do_preprocess:
        preprocess_mode = st.selectbox("Preprocessing Mode", ["Standard", "Enhanced", "Aggressive"])
    else:
        preprocess_mode = None
    
    st.header("LLM Processing")
    api_key = st.text_input("OpenAI API Key", type="password")
    if api_key:
        st.success("API Key provided")
    else:
        st.info("Enter your OpenAI API key to enable AI processing of OCR results")
    
    llm_model = st.selectbox("LLM Model", ["gpt-3.5-turbo", "gpt-4"], index=0)
    
    with st.expander("Advanced"):
        show_intermediate = st.checkbox("Show Processing Steps", value=True)
        dpi = st.slider("PDF DPI", 100, 600, 300, 50)

# Tabs
tab1, tab2 = st.tabs(["Image Upload", "PDF Upload"])

with tab1:
    st.subheader("Upload Image")
    uploaded_image = st.file_uploader("Choose image", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded_image:
        pil_img = Image.open(io.BytesIO(uploaded_image.read()))
        cv_img = pil_to_cv2(pil_img)
        st.image(pil_img, caption="Original", use_container_width=True)
        process_col1, process_col2 = st.columns([1, 1])
        
        with process_col1:
            if st.button("Process Image"):
                with st.spinner("Processing..."):
                    res = process_image_with_cropping(cv_img, enable_cropping, do_preprocess, preprocess_mode)
                    st.session_state['ocr_results'] = res
                    st.session_state['processed'] = True
                    st.success("Image processed")
        
        # Only show LLM button if API key is provided
        with process_col2:
            llm_button_disabled = not api_key
            if st.button("Process Image & Analyze with LLM", disabled=llm_button_disabled):
                with st.spinner("Processing and analyzing..."):
                    res = process_image_with_cropping(cv_img, enable_cropping, do_preprocess, preprocess_mode)
                    st.session_state['ocr_results'] = res
                    st.session_state['processed'] = True
                    
                    # Run OCR
                    _, text = run_ocr(res["cropped"])
                    st.session_state['ocr_text'] = text
                    
                    # Process with LLM
                    llm_result = process_with_llm(text, api_key, llm_model)
                    st.session_state['llm_result'] = llm_result
                    
                    st.success("Image processed and analyzed")
        
        # Display results if processing has been done
        if 'processed' in st.session_state and st.session_state['processed']:
            res = st.session_state['ocr_results']
            
            # Run OCR if not already done
            if 'ocr_text' not in st.session_state:
                _, text = run_ocr(res["cropped"])
                st.session_state['ocr_text'] = text
            else:
                text = st.session_state['ocr_text']
            
            st.subheader("Extracted Text")
            text_area = st.text_area("OCR Output", text, height=200)
            
            # Update session state if text was edited
            if text_area != text:
                st.session_state['ocr_text'] = text_area
            
            # Separate LLM analysis button
            if api_key and ('llm_result' not in st.session_state) and st.button("Analyze with LLM"):
                with st.spinner("Analyzing with OpenAI..."):
                    llm_result = process_with_llm(text_area, api_key, llm_model)
                    st.session_state['llm_result'] = llm_result
                    st.rerun()
            
            # Display LLM results if available
            if 'llm_result' in st.session_state and st.session_state['llm_result']:
                llm_result = st.session_state['llm_result']
                st.subheader("LLM Analysis")
                
                if "error" in llm_result:
                    st.error(f"LLM processing error: {llm_result['error']}")
                    if "details" in llm_result:
                        with st.expander("Error Details"):
                            st.code(llm_result["details"])
                else:
                    # Display the structured data
                    with st.expander("Raw JSON Output", expanded=False):
                        st.json(llm_result)
                    
                    # Also show in a more user-friendly format if it has expected fields
                    if isinstance(llm_result, dict):
                        with st.expander("Formatted Receipt Information", expanded=True):
                            col1, col2 = st.columns(2)
                            
                            # Left column for general info
                            with col1:
                                st.subheader("General Information")
                                for key in ["Store", "Merchant", "Company", "Date", "Total", "Tax", "Payment"]:
                                    for field in llm_result:
                                        if key.lower() in field.lower() and llm_result[field]:
                                            st.write(f"**{field}:** {llm_result[field]}")
                            
                            # Right column for items
                            with col2:
                                st.subheader("Items")
                                # Look for items array with different possible keys
                                item_fields = ["Items", "items", "LineItems", "line_items", "Products", "products"]
                                for item_field in item_fields:
                                    if item_field in llm_result and isinstance(llm_result[item_field], list):
                                        for item in llm_result[item_field]:
                                            if isinstance(item, dict):
                                                item_name = item.get("name", item.get("description", "Unknown Item"))
                                                item_price = item.get("price", item.get("amount", "N/A"))
                                                st.write(f"- {item_name}: {item_price}")
                                            else:
                                                st.write(f"- {item}")
                                        break

            if show_intermediate:
                st.subheader("Processing Steps")
                cols = st.columns(3)
                if "bbox" in res:
                    cols[0].image(cv2_to_pil(res["bbox"]), caption="Detection", use_container_width=True)
                if "preprocessed" in res:
                    img = res["preprocessed"]
                    cols[1].image(Image.fromarray(img), caption="Preprocessed", use_container_width=True)
                if "cropped_preprocessed" in res:
                    img = res["cropped_preprocessed"]
                    cols[2].image(Image.fromarray(img), caption="Cropped Preprocessed", use_container_width=True)

with tab2:
    st.subheader("Upload PDF")
    uploaded_pdf = st.file_uploader("Choose PDF", type=["pdf"])
    if uploaded_pdf:
        st.write(f"Uploaded: {uploaded_pdf.name}")
        process_col1, process_col2 = st.columns([1, 1])
        
        with process_col1:
            if st.button("Process PDF"):
                st.session_state['pdf_processed'] = False  # Reset
                with st.spinner("Processing..."):
                    try:
                        pdf_bytes = uploaded_pdf.getvalue()
                        images = convert_from_bytes(pdf_bytes, dpi=dpi)
                        if images:
                            first_page = images[0]
                            st.image(first_page, caption="Page 1", use_container_width=True)
                            cv_img = pil_to_cv2(first_page)
                            res = process_image_with_cropping(cv_img, enable_cropping, do_preprocess, preprocess_mode)
                            
                            # Store results in session state
                            st.session_state['pdf_results'] = res
                            st.session_state['pdf_image'] = first_page
                            
                            # Run OCR
                            _, text = run_ocr(res["cropped"])
                            st.session_state['pdf_ocr_text'] = text
                            
                            st.session_state['pdf_processed'] = True
                            st.success("PDF page processed")
                        else:
                            st.error("No images extracted from PDF")
                    except Exception as e:
                        st.error(f"PDF processing error: {e}")
                        st.info("For PDF support, install poppler.\n- Windows: http://blog.alivate.com.au/poppler-windows/\n- macOS: brew install poppler\n- Linux: apt install poppler-utils")
        
        # Only show LLM button if API key is provided
        with process_col2:
            llm_button_disabled = not api_key
            if st.button("Process PDF & Analyze with LLM", disabled=llm_button_disabled):
                st.session_state['pdf_processed'] = False  # Reset
                with st.spinner("Processing and analyzing..."):
                    try:
                        pdf_bytes = uploaded_pdf.getvalue()
                        images = convert_from_bytes(pdf_bytes, dpi=dpi)
                        if images:
                            first_page = images[0]
                            st.image(first_page, caption="Page 1", use_container_width=True)
                            cv_img = pil_to_cv2(first_page)
                            res = process_image_with_cropping(cv_img, enable_cropping, do_preprocess, preprocess_mode)
                            
                            # Store results in session state
                            st.session_state['pdf_results'] = res
                            st.session_state['pdf_image'] = first_page
                            
                            # Run OCR
                            _, text = run_ocr(res["cropped"])
                            st.session_state['pdf_ocr_text'] = text
                            
                            # Process with LLM
                            llm_result = process_with_llm(text, api_key, llm_model)
                            st.session_state['pdf_llm_result'] = llm_result
                            
                            st.session_state['pdf_processed'] = True
                            st.success("PDF processed and analyzed")
                        else:
                            st.error("No images extracted from PDF")
                    except Exception as e:
                        st.error(f"PDF processing error: {e}")
                        st.info("For PDF support, install poppler.\n- Windows: http://blog.alivate.com.au/poppler-windows/\n- macOS: brew install poppler\n- Linux: apt install poppler-utils")
        
        # Display results if processing has been done
        if 'pdf_processed' in st.session_state and st.session_state['pdf_processed']:
            # Get results from session state
            res = st.session_state['pdf_results']
            text = st.session_state['pdf_ocr_text']
            
            st.subheader("Extracted Text")
            text_area = st.text_area("PDF OCR Output", text, height=200)
            
            # Update session state if text was edited
            if text_area != text:
                st.session_state['pdf_ocr_text'] = text_area
            
            # Separate LLM analysis button
            if api_key and ('pdf_llm_result' not in st.session_state) and st.button("Analyze PDF with LLM"):
                with st.spinner("Analyzing with OpenAI..."):
                    llm_result = process_with_llm(text_area, api_key, llm_model)
                    st.session_state['pdf_llm_result'] = llm_result
                    st.rerun()
            
            # Display LLM results if available
            if 'pdf_llm_result' in st.session_state and st.session_state['pdf_llm_result']:
                llm_result = st.session_state['pdf_llm_result']
                st.subheader("LLM Analysis")
                
                if "error" in llm_result:
                    st.error(f"LLM processing error: {llm_result['error']}")
                    if "details" in llm_result:
                        with st.expander("Error Details"):
                            st.code(llm_result["details"])
                else:
                    # Display the structured data
                    with st.expander("Raw JSON Output", expanded=False):
                        st.json(llm_result)
                    
                    # Also show in a more user-friendly format if it has expected fields
                    if isinstance(llm_result, dict):
                        with st.expander("Formatted Receipt/Invoice Information", expanded=True):
                            col1, col2 = st.columns(2)
                            
                            # Left column for general info
                            with col1:
                                st.subheader("General Information")
                                for key in ["Store", "Merchant", "Company", "Invoice", "Date", "Total", "Tax", "Payment"]:
                                    for field in llm_result:
                                        if key.lower() in field.lower() and llm_result[field]:
                                            st.write(f"**{field}:** {llm_result[field]}")
                            
                            # Right column for items
                            with col2:
                                st.subheader("Items")
                                # Look for items array with different possible keys
                                item_fields = ["Items", "items", "LineItems", "line_items", "Products", "products"]
                                for item_field in item_fields:
                                    if item_field in llm_result and isinstance(llm_result[item_field], list):
                                        for item in llm_result[item_field]:
                                            if isinstance(item, dict):
                                                item_name = item.get("name", item.get("description", "Unknown Item"))
                                                item_price = item.get("price", item.get("amount", "N/A"))
                                                st.write(f"- {item_name}: {item_price}")
                                            else:
                                                st.write(f"- {item}")
                                        break
            
            if show_intermediate:
                st.subheader("Processing Steps")
                cols = st.columns(3)
                if "bbox" in res:
                    cols[0].image(cv2_to_pil(res["bbox"]), caption="Detection", use_container_width=True)
                if "preprocessed" in res:
                    img = res["preprocessed"]
                    cols[1].image(Image.fromarray(img), caption="Preprocessed", use_container_width=True)
                if "cropped_preprocessed" in res:
                    img = res["cropped_preprocessed"]
                    cols[2].image(Image.fromarray(img), caption="Cropped Preprocessed", use_container_width=True)

st.markdown("---")
st.caption("Receipt OCR App © 2024")