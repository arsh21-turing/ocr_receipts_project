# final_app.py - A Streamlit application for PaddleOCR with LLM-enhanced receipt parsing (improved security)

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
import time
import re
import logging
from datetime import datetime
import pandas as pd

# ---------------------------
# 1. Configure Logging and Currency Support
# ---------------------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "DEBUG").upper()
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_FILE = os.environ.get("LOG_FILE", "receipt_processor.log")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

receipt_logger = logging.getLogger("receipt_processor")
receipt_logger.setLevel(getattr(logging, LOG_LEVEL))
receipt_logger.info("Receipt Processor initialized with LOG_LEVEL=%s", LOG_LEVEL)

# Define currency patterns for detection
CURRENCY_PATTERNS = {
    "$": {"name": "US Dollar", "locale": "en_US", "regex": r"\$\s*\d+\.\d{2}"},
    "€": {"name": "Euro", "locale": "en_EU", "regex": r"€\s*\d+[.,]\d{2}|\d+[.,]\d{2}\s*€"},
    "£": {"name": "British Pound", "locale": "en_GB", "regex": r"£\s*\d+\.\d{2}|\d+\.\d{2}\s*£"},
    "¥": {"name": "Japanese Yen", "locale": "ja_JP", "regex": r"¥\s*\d+|\d+\s*¥"},
    "₹": {"name": "Indian Rupee", "locale": "hi_IN", "regex": r"₹\s*\d+[.,]\d{2}|\d+[.,]\d{2}\s*₹"},
    "₽": {"name": "Russian Ruble", "locale": "ru_RU", "regex": r"₽\s*\d+[.,]\d{2}|\d+[.,]\d{2}\s*₽"}
}
receipt_logger.debug("Loaded currency patterns: %s", list(CURRENCY_PATTERNS.keys()))

def detect_currency(text_lines):
    """
    Detect currency from OCR text lines.
    Returns a tuple of (currency_symbol, locale).
    """
    receipt_logger.debug("Starting currency detection from %d lines", len(text_lines))
    full_text = " ".join(text_lines)
    currency_counts = {}
    for symbol, data in CURRENCY_PATTERNS.items():
        matches = re.findall(data["regex"], full_text)
        if matches:
            currency_counts[symbol] = len(matches)
            receipt_logger.debug("Found %d occurrences of %s (%s)", len(matches), symbol, data["name"])
    if currency_counts:
        most_common = max(currency_counts.items(), key=lambda x: x[1])
        receipt_logger.info("Detected currency: %s (%s) with %d occurrences", 
                            most_common[0], CURRENCY_PATTERNS[most_common[0]]["name"], most_common[1])
        return most_common[0], CURRENCY_PATTERNS[most_common[0]]["locale"]
    receipt_logger.warning("No currency detected, using default: $")
    return "$", "en_US"

# ---------------------------
# 2. Helper Functions for Receipt Extraction & JSON Output
# ---------------------------
def extract_address_from_line(line):
    """
    Determine if a line contains address information.
    """
    address_indicators = ['street', 'st.', 'avenue', 'ave', 'road', 'rd', 'boulevard', 'blvd', 
                          'drive', 'dr', 'lane', 'ln', 'place', 'pl', 'court', 'ct', 
                          'circle', 'way', 'parkway', 'pkwy', 'highway', 'hwy', 'suite', 'ste',
                          'floor', 'fl', 'apartment', 'apt', 'unit', 'room', 'rm']
    postal_patterns = [r'\b\d{5}(?:-\d{4})?\b', r'\b[A-Z]\d[A-Z]\s*\d[A-Z]\d\b', r'\b[A-Z]{1,2}\d{1,2}\s*\d[A-Z]{2}\b']
    state_indicators = [r'\b[A-Z]{2}\b', r',\s*[A-Z]{2}']
    has_address_indicator = any(indicator in line.lower() for indicator in address_indicators)
    has_postal_code = any(re.search(pattern, line) for pattern in postal_patterns)
    has_state_indicator = any(re.search(pattern, line) for pattern in state_indicators)
    has_address_pattern = (re.search(r'\d+\s+[A-Za-z]', line) or re.search(r'(?:Suite|Ste|Unit|Apt)\.?\s*#?\s*\d+', line, re.IGNORECASE))
    return has_address_indicator or has_postal_code or has_state_indicator or has_address_pattern

def extract_receipt_information(ocr_results):
    """
    Extract structured information from receipt OCR results using pattern matching.
    """
    receipt_logger.info("Starting receipt information extraction from %d OCR results", len(ocr_results))
    receipt_data = {
        'store_name': '',
        'date': '',
        'time': '',
        'address': '',
        'phone': '',
        'receipt_no': '',
        'subtotal': '',
        'tax': '',
        'total': '',
        'payment_method': '',
        'card_info': '',
        'items': [],
        'currency_symbol': '$',
        'locale': 'en_US'
    }
    if not ocr_results or not isinstance(ocr_results, list):
        receipt_logger.warning("Invalid OCR results provided to receipt parser")
        return receipt_data

    # Extract text lines and sort by Y-coordinate (top to bottom)
    text_lines = []
    for result in ocr_results:
        if len(result) < 3:
            receipt_logger.warning("Skipping OCR result due to unexpected format: %s", result)
            continue
        text, conf, box = result[:3]
        try:
            y_center = sum(point[1] for point in box) / len(box)
        except Exception as e:
            receipt_logger.warning("Error calculating y_center for box %s: %s", box, str(e))
            continue
        text_lines.append((text, y_center, conf))
    text_lines.sort(key=lambda x: x[1])
    receipt_logger.info("Sorted %d text lines by vertical position", len(text_lines))
    text_lines_only = [line[0] for line in text_lines]
    for i, line in enumerate(text_lines_only):
        receipt_logger.debug("Line %d: %s", i+1, line)

    # Detect currency from text
    receipt_logger.info("Detecting currency from receipt text")
    currency_symbol, locale = detect_currency(text_lines_only)
    receipt_data['currency_symbol'] = currency_symbol
    receipt_data['locale'] = locale

    receipt_logger.info("Starting information extraction from text lines")
    items_section = False
    address_collection_mode = False
    address_line_count = 0
    MAX_ADDRESS_LINES = 3

    for i, line in enumerate(text_lines_only):
        receipt_logger.debug("Processing line %d: %s", i+1, line[:50])
        # Date extraction
        date_match = re.search(r'(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})', line)
        if date_match and not receipt_data['date']:
            receipt_data['date'] = date_match.group(1)
            receipt_logger.info("Detected date: %s", receipt_data['date'])
        # Time extraction
        time_match = re.search(r'(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?)', line)
        if time_match and not receipt_data['time']:
            receipt_data['time'] = time_match.group(1)
            receipt_logger.info("Detected time: %s", receipt_data['time'])
        # Phone extraction
        phone_match = re.search(r'(?:Tel|Phone|Ph)?\s*[:#]?\s*(\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})', line)
        if phone_match and not receipt_data['phone']:
            receipt_data['phone'] = phone_match.group(1)
            receipt_logger.info("Detected phone: %s", receipt_data['phone'])
        # Total amount extraction
        total_pattern = fr'(?:Total|TOTAL|Amount|Summe|Gesamt)[\s:]*{re.escape(currency_symbol)}?\s*(\d+[.,]\d{{2}})'
        total_match = re.search(total_pattern, line)
        if total_match and not receipt_data['total']:
            receipt_data['total'] = total_match.group(1)
            receipt_logger.info("Detected total: %s %s", currency_symbol, receipt_data['total'])
        # Subtotal extraction
        subtotal_pattern = fr'(?:Subtotal|Sub-total|SUB|Zwischensumme|Sous-total|小計)[\s:]*{re.escape(currency_symbol)}?\s*(\d+[.,]\d{{2}})'
        subtotal_match = re.search(subtotal_pattern, line)
        if subtotal_match and not receipt_data['subtotal']:
            receipt_data['subtotal'] = subtotal_match.group(1)
            receipt_logger.info("Detected subtotal: %s %s", currency_symbol, receipt_data['subtotal'])
        # Tax extraction
        tax_pattern = fr'(?:Tax|TAX|VAT|GST|MwSt|IVA|TVA|消費税|Steuer)[\s:]*{re.escape(currency_symbol)}?\s*(\d+[.,]\d{{2}})'
        tax_match = re.search(tax_pattern, line)
        if tax_match and not receipt_data['tax']:
            receipt_data['tax'] = tax_match.group(1)
            receipt_logger.info("Detected tax: %s %s", currency_symbol, receipt_data['tax'])
        # Receipt number extraction
        receipt_no_pattern = r'(?:Receipt|Order|Invoice|Factura|Rechnung|Quittung|Reçu|Bon|レシート|發票|No\.)[\s#:\.\-_]*(\w+\d+|\d+[\w\-]+|\d+)'
        receipt_no_match = re.search(receipt_no_pattern, line, re.IGNORECASE)
        if receipt_no_match and not receipt_data['receipt_no']:
            receipt_data['receipt_no'] = receipt_no_match.group(1)
            receipt_logger.info("Detected receipt number: %s", receipt_data['receipt_no'])
        # Payment method extraction
        receipt_logger.debug("Checking for payment method in: '%s'", line)
        payment_methods = {
            'Cash': ['CASH', 'BARGELD', 'ESPÈCES', '現金', 'EFECTIVO', 'CONTANT'],
            'Credit': ['CREDIT', 'KREDITKARTE', 'CARTE DE CRÉDIT', 'クレジット', 'CREDITO'],
            'Debit': ['DEBIT', 'EC-KARTE', 'CARTE DE DÉBIT', 'デビット', 'DEBITO'],
            'Visa': ['VISA'],
            'Mastercard': ['MASTERCARD', 'MASTER CARD'],
            'Amex': ['AMEX', 'AMERICAN EXPRESS'],
            'PayPal': ['PAYPAL'],
            'Apple Pay': ['APPLE PAY', 'APPLEPAY'],
            'Google Pay': ['GOOGLE PAY', 'GOOGLEPAY', 'G PAY'],
            'WeChat Pay': ['WECHAT PAY', 'WECHATPAY', '微信支付'],
            'Alipay': ['ALIPAY', 'ALIBABA PAY', '支付宝']
        }
        payment_detected = False
        for payment_type, terms in payment_methods.items():
            for term in terms:
                if term in line.upper() and not receipt_data['payment_method']:
                    receipt_data['payment_method'] = payment_type
                    receipt_logger.info("Detected payment method: %s", payment_type)
                    payment_detected = True
                    break
            if payment_detected:
                break
        # Store name extraction
        if not receipt_data['store_name'] and len(text_lines_only) > 0 and line == text_lines_only[0]:
            receipt_data['store_name'] = line
            receipt_logger.info("Detected store name (first line): %s", receipt_data['store_name'])
        if not receipt_data['store_name'] and i < 5:
            store_indicators = ['store', 'shop', 'restaurant', 'café', 'cafe', 'market', 'supermarket', 'ltd', 'inc', 'gmbh', 'co.', 'ag']
            for indicator in store_indicators:
                if indicator.lower() in line.lower():
                    receipt_data['store_name'] = line
                    receipt_logger.info("Detected store name from indicator '%s': %s", indicator, line)
                    break
        # Enhanced address extraction
        if receipt_data['store_name']:
            is_address_line = extract_address_from_line(line)
            if is_address_line:
                if not address_collection_mode:
                    address_collection_mode = True
                    receipt_data['address_lines'] = []
                    receipt_logger.info("Starting address collection with: %s", line)
                if address_line_count < MAX_ADDRESS_LINES:
                    receipt_data['address_lines'].append(line)
                    receipt_logger.debug("Added address line %d: %s", address_line_count + 1, line)
                    address_line_count += 1
                    if not receipt_data.get('address'):
                        receipt_data['address'] = line
                        receipt_logger.info("Primary address set to: %s", line)
                elif address_line_count >= MAX_ADDRESS_LINES:
                    address_collection_mode = False
                    receipt_logger.debug("Address collection complete with %d lines", address_line_count)
            else:
                if address_collection_mode:
                    address_collection_mode = False
                    receipt_logger.debug("Address collection ended after %d lines", address_line_count)
        # Items section detection
        try:
            item_indicators = r'(ITEM|QTY|QUANTITY|PRICE|AMOUNT|DESCRIPTION|ARTIKEL|ANZAHL|PREIS|BETRAG|QUANTITÉ|PRIX|MONTANT|DESCRIPCIÓN|CANTIDAD|PRECIO|個数|単価|金額|品目)'
            if re.search(item_indicators, line.upper()):
                items_section = True
                receipt_logger.info("Detected start of items section at line %d: %s", i+1, line)
                continue
        except re.error as e:
            receipt_logger.error("Error in regex pattern for items section detection: %s", e)
        # Process items
        if items_section:
            try:
                price_pattern = fr'([{re.escape(currency_symbol)}]?\s*\d+[.,]\d{{2}})'
                price_match = re.search(price_pattern, line)
                if price_match:
                    price = price_match.group(1)
                    item_name = line.replace(price, '').strip()
                    receipt_logger.debug("Found potential item: '%s' with price '%s'", item_name, price)
                    qty_match = re.search(r'(\d+)\s*(?:x|\@|×)', item_name, re.IGNORECASE)
                    quantity = qty_match.group(1) if qty_match else '1'
                    if item_name:
                        clean_price = price.replace(currency_symbol, '').strip()
                        if ',' in clean_price and '.' not in clean_price:
                            clean_price = clean_price.replace(',', '.')
                        receipt_data['items'].append({
                            'item': item_name,
                            'price': clean_price,
                            'quantity': quantity,
                            'currency': currency_symbol
                        })
                        receipt_logger.info("Added item: %s, Price: %s, Qty: %s", item_name, clean_price, quantity)
            except re.error as e:
                receipt_logger.error("Error in regex pattern for item price: %s", e)
    
    receipt_logger.info("Receipt information extraction complete")
    receipt_logger.info("Found %d items in receipt", len(receipt_data['items']))
    for key, value in receipt_data.items():
        if key != 'items':
            receipt_logger.debug("Receipt data - %s: %s", key, value)
    
    # Validate totals if possible
    if receipt_data['subtotal'] and receipt_data['tax'] and receipt_data['total']:
        try:
            subtotal_val = float(receipt_data['subtotal'].replace(',', '.'))
            tax_val = float(receipt_data['tax'].replace(',', '.'))
            total_val = float(receipt_data['total'].replace(',', '.'))
            calculated_total = subtotal_val + tax_val
            if abs(calculated_total - total_val) < 0.02:
                receipt_logger.info("Total validation successful: %.2f + %.2f = %.2f", subtotal_val, tax_val, total_val)
            else:
                receipt_logger.warning("Total validation discrepancy: %.2f + %.2f = %.2f, receipt shows %.2f", subtotal_val, tax_val, calculated_total, total_val)
        except ValueError:
            receipt_logger.error("Error converting monetary values for validation")
    
    receipt_logger.debug("Final extracted receipt data: %s", receipt_data)
    return receipt_data

def prepare_receipt_json(receipt_data, ocr_results=None, source_filename=None):
    """
    Prepare enriched JSON representation of receipt data.
    """
    receipt_logger.debug("Preparing JSON output for receipt data")
    json_data = receipt_data.copy()
    json_data["metadata"] = {
        "timestamp": datetime.now().isoformat(),
        "extracted_at": datetime.now().isoformat(),
        "version": "1.0"
    }
    if source_filename:
        json_data["metadata"]["source"] = source_filename
    if ocr_results:
        confidence_scores = []
        for res in ocr_results:
            if len(res) >= 3:
                try:
                    score = float(res[1])
                    confidence_scores.append(score)
                except ValueError:
                    receipt_logger.warning("Non-numeric confidence encountered: %s", res[1])
        if confidence_scores:
            json_data["metadata"]["ocr_stats"] = {
                "confidence_avg": sum(confidence_scores) / len(confidence_scores),
                "confidence_min": min(confidence_scores),
                "confidence_max": max(confidence_scores),
                "text_elements": len(ocr_results)
            }
    if json_data.get('address_lines'):
        json_data['address_full'] = '\n'.join(json_data['address_lines'])
        receipt_logger.debug("Combined address lines into full address")
        try:
            if len(json_data['address_lines']) >= 2:
                address_components = {}
                last_line = json_data['address_lines'][-1]
                city_state_zip = re.search(r'([A-Za-z\s]+),\s*([A-Z]{2})\s*(\d{5}(?:-\d{4})?)', last_line)
                if city_state_zip:
                    address_components['city'] = city_state_zip.group(1).strip()
                    address_components['state'] = city_state_zip.group(2)
                    address_components['postal_code'] = city_state_zip.group(3)
                    if len(json_data['address_lines']) > 1:
                        address_components['street'] = json_data['address_lines'][0]
                    json_data['address_components'] = address_components
                    receipt_logger.info("Structured address components extracted: %s", address_components)
        except Exception as e:
            receipt_logger.warning("Could not structure address components: %s", str(e))
    if json_data.get('date'):
        try:
            date_parts = re.findall(r'\d+', json_data['date'])
            if len(date_parts) >= 3:
                if int(date_parts[0]) <= 31 and int(date_parts[1]) <= 12:
                    year = date_parts[2]
                    if len(year) == 2:
                        year = "20" + year
                    iso_date = f"{year}-{date_parts[1].zfill(2)}-{date_parts[0].zfill(2)}"
                    json_data["date_iso"] = iso_date
        except Exception as e:
            receipt_logger.warning("Could not standardize date format: %s", e)
    if json_data.get('items'):
        for item in json_data['items']:
            if item.get('price'):
                try:
                    item['price'] = f"{float(item['price']):.2f}"
                except ValueError:
                    receipt_logger.warning("Could not convert item price to float: %s", item.get('price'))
            if item.get('quantity'):
                try:
                    item['quantity'] = int(item['quantity'])
                except ValueError:
                    pass
    if not json_data.get('currency_symbol'):
        json_data['currency_symbol'] = '$'
    receipt_logger.debug("JSON preparation complete with %d fields", len(json_data))
    return json_data

def dump_receipt_to_terminal(receipt_data):
    """Print receipt data to terminal for debugging"""
    receipt_logger.info("=== RECEIPT DATA DUMP ===")
    for key, value in receipt_data.items():
        if key != 'items':
            receipt_logger.info("%s: %s", key.upper(), value)
    if receipt_data['items']:
        receipt_logger.info("ITEMS:")
        for i, item in enumerate(receipt_data['items']):
            receipt_logger.info("  %d. %s - Qty: %s, Price: %s%s", i+1, item['item'], item['quantity'], receipt_data['currency_symbol'], item['price'])
    receipt_logger.info("=== END RECEIPT DUMP ===")

def log_extraction_stats(ocr_results, receipt_data, elapsed_time):
    """Log statistics about the extraction process"""
    receipt_logger.info("OCR Extraction Statistics:")
    receipt_logger.info("- Processed %d OCR text elements", len(ocr_results))
    receipt_logger.info("- Detected %d items", len(receipt_data['items']))
    receipt_logger.info("- Extraction completed in %.2f seconds", elapsed_time)
    simple_fields = ['store_name', 'date', 'total']
    confidence_calc = {field: bool(receipt_data[field]) for field in simple_fields}
    confidence_score = sum(confidence_calc.values()) / len(confidence_calc)
    receipt_logger.info("- Extraction confidence: %.2f", confidence_score)
    if confidence_score < 0.5:
        receipt_logger.warning("Low confidence extraction - check results carefully")

# ---------------------------
# 3. Cropping, Preprocessing and OCR Functions
# ---------------------------
from cropping_pipeline import CroppingPipeline
# Replace the current import section in the main file with this:

try:
    from grayscale_processor import GrayscaleProcessor
    
    # Create a preprocessor instance
    grayscale_processor = GrayscaleProcessor()
    
    def preprocess_image(image, adaptive=True):
        """Wrapper function to use the GrayscaleProcessor class"""
        if adaptive:
            return grayscale_processor.process_for_receipt_ocr(image)
        else:
            # Use Otsu thresholding for non-adaptive processing
            gray = grayscale_processor.to_grayscale(image)
            return grayscale_processor.apply_threshold(gray, method='otsu')
            
except ImportError:
    receipt_logger.warning("Could not import GrayscaleProcessor class, using fallback implementation")
    
    def preprocess_image(image, adaptive=True):
        """Fallback preprocessing function"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        if adaptive:
            return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        else:
            _, processed = cv2.threshold(gray, 0, 255,
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return processed

# Define helper functions to convert between PIL and OpenCV images
def pil_to_cv2(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_image):
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

from paddleocr import PaddleOCR
ocr_engine = PaddleOCR(use_angle_cls=True, lang='en')

@st.cache_resource
def get_cropping_pipeline():
    return CroppingPipeline()

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
            receipt_logger.error("Cropping error: %s", str(e), exc_info=True)
            results["cropped"] = image
            results["cropped_preprocessed"] = working_image
    else:
        results["cropped"] = image
        results["cropped_preprocessed"] = working_image
    return results

# ---------------------------
# 4. LLM-Enhanced Receipt Parsing Functions
def process_with_llm(text, api_key, model="gpt-3.5-turbo"):
    """
    Process OCR text with OpenAI language model to extract structured receipt data.
    
    Args:
        text: OCR text from receipt
        api_key: OpenAI API key
        model: LLM model to use
        
    Returns:
        dict: Structured receipt data as JSON
    """
    if not api_key or not text:
        return None
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        prompt = f"""
        Extract structured data from this receipt text. 
        If it's a receipt, extract:
        - Store/Merchant name
        - Date
        - Time (if available)
        - Total amount
        - Subtotal (if available)
        - Individual items with prices if available
        - Tax amount if available
        - Payment method if available
        - Receipt number (if available)
        - Store address (if available)
        
        Format the response as a clean JSON object without any explanation text.
        If a field is not found, omit it from the JSON entirely rather than including empty values.
        Each item should include at least name and price fields.
        
        Text from receipt:
        {text}
        """
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a receipt data extraction assistant. Return clean, structured JSON only without any explanation or markdown formatting."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3
        }
        
        # Add response_format parameter for compatible models (GPT-4 and newer GPT-3.5)
        if "gpt-4" in model or "gpt-3.5" in model:
            payload["response_format"] = {"type": "json_object"}
        
        receipt_logger.debug(f"Sending request to LLM API with model {model}")
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            response_data = response.json()
            result = response_data['choices'][0]['message']['content']
            receipt_logger.debug("Received response from LLM")
            
            # Clean up the result to handle possible markdown formatting
            result = result.strip()
            
            # Remove markdown code blocks if present
            if result.startswith("```json"):
                result = result[7:]
            elif result.startswith("```"):
                result = result[3:]
            
            if result.endswith("```"):
                result = result[:-3]
            
            result = result.strip()
            receipt_logger.debug(f"Cleaned result: {result[:100]}...")
            
            try:
                json_result = json.loads(result)
                receipt_logger.info("Successfully parsed JSON from LLM response")
                
                # Process the items list if it exists
                if "items" in json_result and isinstance(json_result["items"], list):
                    for item in json_result["items"]:
                        if isinstance(item, dict) and "price" in item:
                            # Clean price string (remove currency symbols, normalize decimals)
                            price_str = str(item["price"])
                            for symbol in ["$", "€", "£", "¥", "₹", "₽"]:
                                price_str = price_str.replace(symbol, "")
                            
                            # Handle comma as decimal separator
                            if "," in price_str and "." not in price_str:
                                price_str = price_str.replace(",", ".")
                            
                            try:
                                item["price"] = float(price_str.strip())
                                item["price"] = f"{item['price']:.2f}"  # Format as string with 2 decimal places
                            except ValueError:
                                # Keep as is if conversion fails
                                receipt_logger.warning(f"Could not convert price to float: {price_str}")
                
                # Format monetary values consistently if they exist
                for field in ["total", "subtotal", "tax"]:
                    if field in json_result and isinstance(json_result[field], str):
                        price_str = json_result[field]
                        for symbol in ["$", "€", "£", "¥", "₹", "₽"]:
                            price_str = price_str.replace(symbol, "")
                        
                        if "," in price_str and "." not in price_str:
                            price_str = price_str.replace(",", ".")
                            
                        try:
                            json_result[field] = float(price_str.strip())
                            json_result[field] = f"{json_result[field]:.2f}"
                        except ValueError:
                            receipt_logger.warning(f"Could not convert {field} to float: {price_str}")
                
                # Add metadata
                json_result["metadata"] = {
                    "timestamp": datetime.now().isoformat(),
                    "extracted_with": "llm",
                    "model": model,
                    "version": "1.0"
                }
                
                return json_result
                
            except json.JSONDecodeError as e:
                receipt_logger.error(f"Failed to parse JSON from LLM: {str(e)}")
                
                # Try to find a JSON object in the response
                import re
                json_pattern = r'({[^{}]*(?:{[^{}]*}[^{}]*)*})'
                json_matches = re.findall(json_pattern, result)
                
                if json_matches:
                    for potential_json in json_matches:
                        try:
                            json_result = json.loads(potential_json)
                            receipt_logger.info("Found valid JSON object in response")
                            return json_result
                        except:
                            continue
                
                # Return the raw response if we can't parse it
                return {
                    "error": "Invalid JSON format in LLM response",
                    "raw_response": result
                }
        else:
            receipt_logger.error(f"API error: {response.status_code}")
            return {
                "error": f"API Error: {response.status_code}",
                "details": response.text
            }
    except Exception as e:
        receipt_logger.error(f"Exception in LLM processing: {str(e)}", exc_info=True)
        return {"error": str(e)}

def enhance_receipt_data(receipt_data, filtered_results, api_key=None):
    """
    Enhance receipt data extracted by pattern matching with LLM processing.
    """
    if not api_key or api_key.strip() == "":
        return receipt_data
    # Check if pattern matching already extracted useful data
    has_useful_data = (
        bool(receipt_data.get('store_name')) and
        bool(receipt_data.get('total')) and
        len(receipt_data.get('items', [])) > 1
    )
    if has_useful_data:
        return receipt_data
    if filtered_results:
        ocr_text = "\n".join([text for text, _, _ in filtered_results])
    else:
        return receipt_data
    llm_receipt_data = process_with_llm(ocr_text, api_key)
    if llm_receipt_data:
        merged_receipt_data = llm_receipt_data.copy()
        for key in receipt_data:
            if key != 'items':
                if not llm_receipt_data.get(key) and receipt_data.get(key):
                    merged_receipt_data[key] = receipt_data[key]
        if receipt_data.get('items') and not llm_receipt_data.get('items'):
            merged_receipt_data['items'] = receipt_data['items']
        return merged_receipt_data
    return receipt_data

def format_receipt_data_for_display(receipt_data, filtered_results, api_key=None):
    """
    Create a structured representation of receipt data for UI display.
    
    Args:
        receipt_data: Dictionary containing extracted receipt data.
        filtered_results: Original filtered OCR results (for fallback).
        api_key: Optional OpenAI API key for LLM enhancement.
        
    Returns:
        tuple: (receipt_data, has_valid_data, used_llm, items_df)
    """
    has_data = any([
        receipt_data['store_name'],
        receipt_data['date'], 
        receipt_data['total'], 
        receipt_data['items']
    ])
    
    used_llm = False
    if api_key and (not has_data or len(receipt_data.get('items', [])) <= 1):
        try:
            enhanced_data = enhance_receipt_data(receipt_data, filtered_results, api_key)
            if enhanced_data:
                receipt_data = enhanced_data
                used_llm = True
                has_data = any([
                    receipt_data['store_name'],
                    receipt_data['date'],
                    receipt_data['total'],
                    receipt_data['items']
                ])
        except Exception as e:
            logging.error(f"LLM enhancement failed: {str(e)}")
    if not has_data and filtered_results:
        receipt_data = {
            'store_name': 'Unknown Store',
            'date': '',
            'time': '',
            'address': '',
            'phone': '',
            'receipt_no': '',
            'subtotal': '',
            'tax': '',
            'total': '',
            'payment_method': '',
            'card_info': '',
            'items': [],
        }
        for text, _, _ in filtered_results:
            price_match = re.search(r'(\$?\d+\.\d{2})', text)
            if price_match:
                price = price_match.group(1)
                item_name = text.replace(price, '').strip()
                if item_name:
                    receipt_data['items'].append({
                        'item': item_name,
                        'price': price.replace('$', ''),
                        'quantity': '1'
                    })
    items_df = None
    if receipt_data['items']:
        items_df = pd.DataFrame(receipt_data['items'])
        if 'price' in items_df.columns:
            items_df['price'] = items_df['price'].apply(lambda x: f"${x}")
    return receipt_data, has_data, used_llm, items_df

def generate_formatted_receipt_text(receipt_data):
    """
    Generate formatted text representation of receipt data.
    
    Args:
        receipt_data: Dictionary containing structured receipt information.
        
    Returns:
        str: Formatted receipt text.
    """
    lines = []
    if receipt_data['store_name']:
        lines.append(receipt_data['store_name'].center(50))
        lines.append("")
    if receipt_data['address']:
        lines.append(receipt_data['address'].center(50))
        lines.append("")
    if receipt_data['phone']:
        lines.append(f"Phone: {receipt_data['phone']}".center(50))
        lines.append("")
    date_time_line = ""
    if receipt_data['date'] and receipt_data['time']:
        date_time_line = f"Date: {receipt_data['date']} Time: {receipt_data['time']}"
    elif receipt_data['date']:
        date_time_line = f"Date: {receipt_data['date']}"
    elif receipt_data['time']:
        date_time_line = f"Time: {receipt_data['time']}"
    if date_time_line:
        lines.append(date_time_line.center(50))
        lines.append("")
    if receipt_data['receipt_no']:
        lines.append(f"Receipt #: {receipt_data['receipt_no']}".center(50))
        lines.append("")
    lines.append("-" * 50)
    if receipt_data['items']:
        lines.append("ITEM                  QTY    PRICE")
        lines.append("-" * 50)
        for item in receipt_data['items']:
            item_name = item['item'][:20].ljust(20)
            qty = str(item['quantity']).rjust(5)
            price = item['price'].rjust(10)
            lines.append(f"{item_name} {qty} {price}")
        lines.append("-" * 50)
    if receipt_data['subtotal']:
        lines.append(f"Subtotal: {receipt_data['subtotal']}".rjust(50))
    if receipt_data['tax']:
        lines.append(f"Tax: {receipt_data['tax']}".rjust(50))
    if receipt_data['total']:
        lines.append(f"TOTAL: {receipt_data['total']}".rjust(50))
        lines.append("")
    if receipt_data['payment_method']:
        lines.append(f"Payment Method: {receipt_data['payment_method']}".center(50))
    if receipt_data['card_info']:
        lines.append(f"Card Info: {receipt_data['card_info']}".center(50))
    lines.append("")
    lines.append("Thank you for your business!".center(50))
    lines.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S").center(50))
    return "\n".join(lines)

# ---------------------------
# 5. Streamlit UI Components and Main App
# ---------------------------
def main():
    st.set_page_config(
        page_title="Receipt OCR & Parsing",
        page_icon="📝",
        layout="wide"
    )
    st.title("Receipt OCR & Parsing Tool")
    st.markdown("""
    This application extracts and parses text from receipt images and PDFs using OCR.
    Upload a receipt image or PDF and get structured receipt information.
    """)
    
    # Sidebar Settings
    st.sidebar.title("OCR Settings")
    lang_options = ["en", "ch", "fr", "german", "korean", "japanese"]
    ocr_lang = st.sidebar.selectbox("OCR Language", lang_options, index=0)
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    dpi = st.sidebar.slider("PDF DPI", 100, 600, 300, 50)
    do_preprocess = st.sidebar.checkbox("Preprocess Image", value=True)
    if do_preprocess:
        preprocess_mode = st.sidebar.selectbox("Preprocessing Mode", ["Standard", "Enhanced", "Aggressive"])
    else:
        preprocess_mode = None
    enable_cropping = st.sidebar.checkbox("Enable Auto-Cropping", value=True)
    
    st.header("LLM Processing")
    api_key = st.text_input("OpenAI API Key", type="password")
    if api_key:
        st.success("API Key provided")
    else:
        st.info("Enter your OpenAI API key to enable AI processing of OCR results")
    llm_model = st.selectbox("LLM Model", ["gpt-3.5-turbo", "gpt-4"], index=0)
    with st.expander("Advanced"):
        show_intermediate = st.checkbox("Show Processing Steps", value=True)
    
    tab1, tab2 = st.tabs(["Image Upload", "PDF Upload"])
    
    # Image Upload Tab
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
                        start_time = time.time()
                        res = process_image_with_cropping(cv_img, enable_cropping, do_preprocess, preprocess_mode)
                        st.session_state['ocr_results'] = res
                        st.session_state['processed'] = True
                        elapsed_time = time.time() - start_time
                        receipt_logger.info("Image processed in %.2f seconds", elapsed_time)
                        st.success("Image processed")
            with process_col2:
                llm_button_disabled = not api_key
                if st.button("Process Image & Analyze with LLM", disabled=llm_button_disabled):
                    with st.spinner("Processing and analyzing..."):
                        res = process_image_with_cropping(cv_img, enable_cropping, do_preprocess, preprocess_mode)
                        st.session_state['ocr_results'] = res
                        st.session_state['processed'] = True
                        ocr_result, text = run_ocr(res["cropped"])
                        st.session_state['ocr_text'] = text
                        llm_result = process_with_llm(text, api_key, llm_model)
                        st.session_state['llm_result'] = llm_result
                        st.success("Image processed and analyzed")
            if 'processed' in st.session_state and st.session_state['processed']:
                res = st.session_state['ocr_results']
                if 'ocr_text' not in st.session_state:
                    _, text = run_ocr(res["cropped"])
                    st.session_state['ocr_text'] = text
                else:
                    text = st.session_state['ocr_text']
                st.subheader("Extracted Text")
                text_area = st.text_area("OCR Output", text, height=200)
                if text_area != text:
                    st.session_state['ocr_text'] = text_area
                if api_key and ('llm_result' not in st.session_state) and st.button("Analyze with LLM"):
                    with st.spinner("Analyzing with OpenAI..."):
                        llm_result = process_with_llm(text_area, api_key, llm_model)
                        st.session_state['llm_result'] = llm_result
                        st.rerun()
                if 'llm_result' in st.session_state and st.session_state['llm_result']:
                    llm_result = st.session_state['llm_result']
                    st.subheader("LLM Analysis")
                    
                    # Check for error in LLM result
                    if isinstance(llm_result, dict) and "error" in llm_result:
                        st.error(f"LLM processing error: {llm_result['error']}")
                        if "details" in llm_result:
                            with st.expander("Error Details"):
                                st.code(llm_result["details"])
                        if "raw_response" in llm_result:
                            with st.expander("Raw LLM Response"):
                                st.text(llm_result["raw_response"])
                    else:
                        # Handle case where we have a proper structured result
                        try:
                            # Ensure we have JSON serializable content
                            formatted_json = json.dumps(llm_result, indent=2, default=str)
                            st.code(formatted_json, language="json")
                            
                            # Add download button for the JSON
                            st.download_button(
                                label="Download JSON",
                                data=formatted_json,
                                file_name=f"{uploaded_image.name}_receipt_data.json",
                                mime="application/json"
                            )
                            
                            # Show a receipt summary UI
                            with st.expander("Receipt Summary", expanded=True):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    if "store_name" in llm_result:
                                        st.markdown(f"**Store:** {llm_result['store_name']}")
                                    if "date" in llm_result:
                                        st.markdown(f"**Date:** {llm_result['date']}")
                                    if "receipt_no" in llm_result:
                                        st.markdown(f"**Receipt #:** {llm_result['receipt_no']}")
                                
                                with col2:
                                    if "total" in llm_result:
                                        st.markdown(f"**Total:** {llm_result['total']}")
                                    if "subtotal" in llm_result:
                                        st.markdown(f"**Subtotal:** {llm_result['subtotal']}")
                                    if "tax" in llm_result:
                                        st.markdown(f"**Tax:** {llm_result['tax']}")
                                    if "payment_method" in llm_result:
                                        st.markdown(f"**Payment:** {llm_result['payment_method']}")
                                
                                # Display items as a table if they exist
                                if "items" in llm_result and isinstance(llm_result["items"], list) and len(llm_result["items"]) > 0:
                                    st.markdown("### Items")
                                    
                                    # Convert items to DataFrame for better display
                                    items_df = pd.DataFrame(llm_result["items"])
                                    
                                    # Format price column if it exists
                                    if "price" in items_df.columns:
                                        items_df["price"] = items_df["price"].apply(
                                            lambda x: f"${x}" if not str(x).startswith("$") else x
                                        )
                                    
                                    st.dataframe(items_df, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error displaying LLM results: {str(e)}")
                            st.json(llm_result)  # Fallback to raw JSON display

                st.download_button(
                    label="Download Raw Text",
                    data=st.session_state['ocr_text'],
                    file_name=f"{uploaded_image.name}_ocr_text.txt",
                    mime="text/plain"
                )
    # PDF Upload Tab
    with tab2:
        st.subheader("Upload PDF")
        uploaded_pdf = st.file_uploader("Choose PDF", type=["pdf"])
        if uploaded_pdf:
            st.write(f"Uploaded: {uploaded_pdf.name}")
            process_col1, process_col2 = st.columns([1, 1])
            with process_col1:
                if st.button("Process PDF"):
                    st.session_state['pdf_processed'] = False
                    with st.spinner("Processing..."):
                        try:
                            pdf_bytes = uploaded_pdf.getvalue()
                            images = convert_from_bytes(pdf_bytes, dpi=dpi)
                            if images:
                                first_page = images[0]
                                st.image(first_page, caption="Page 1", use_container_width=True)
                                cv_img = pil_to_cv2(first_page)
                                res = process_image_with_cropping(cv_img, enable_cropping, do_preprocess, preprocess_mode)
                                st.session_state['pdf_results'] = res
                                st.session_state['pdf_image'] = first_page
                                _, text = run_ocr(res["cropped"])
                                st.session_state['pdf_ocr_text'] = text
                                st.session_state['pdf_processed'] = True
                                st.success("PDF page processed")
                            else:
                                st.error("No images extracted from PDF")
                        except Exception as e:
                            st.error(f"PDF processing error: {e}")
            with process_col2:
                llm_button_disabled = not api_key
                if st.button("Process PDF & Analyze with LLM", disabled=llm_button_disabled):
                    st.session_state['pdf_processed'] = False
                    with st.spinner("Processing and analyzing..."):
                        try:
                            pdf_bytes = uploaded_pdf.getvalue()
                            images = convert_from_bytes(pdf_bytes, dpi=dpi)
                            if images:
                                first_page = images[0]
                                st.image(first_page, caption="Page 1", use_container_width=True)
                                cv_img = pil_to_cv2(first_page)
                                res = process_image_with_cropping(cv_img, enable_cropping, do_preprocess, preprocess_mode)
                                st.session_state['pdf_results'] = res
                                st.session_state['pdf_image'] = first_page
                                _, text = run_ocr(res["cropped"])
                                st.session_state['pdf_ocr_text'] = text
                                llm_result = process_with_llm(text, api_key, llm_model)
                                st.session_state['pdf_llm_result'] = llm_result
                                st.session_state['pdf_processed'] = True
                                st.success("PDF processed and analyzed")
                            else:
                                st.error("No images extracted from PDF")
                        except Exception as e:
                            st.error(f"PDF processing error: {e}")
            if 'pdf_processed' in st.session_state and st.session_state['pdf_processed']:
                res = st.session_state['pdf_results']
                text = st.session_state['pdf_ocr_text']
                st.subheader("Extracted Text")
                text_area = st.text_area("PDF OCR Output", text, height=200)
                if text_area != text:
                    st.session_state['pdf_ocr_text'] = text_area
                if api_key and ('pdf_llm_result' not in st.session_state) and st.button("Analyze PDF with LLM"):
                    with st.spinner("Analyzing with OpenAI..."):
                        llm_result = process_with_llm(text_area, api_key, llm_model)
                        st.session_state['pdf_llm_result'] = llm_result
                        st.rerun()
                if 'pdf_llm_result' in st.session_state and st.session_state['pdf_llm_result']:
                    llm_result = st.session_state['pdf_llm_result']
                    st.subheader("LLM Analysis")
                    
                    # Check for error in LLM result
                    if isinstance(llm_result, dict) and "error" in llm_result:
                        st.error(f"LLM processing error: {llm_result['error']}")
                        if "details" in llm_result:
                            with st.expander("Error Details"):
                                st.code(llm_result["details"])
                        if "raw_response" in llm_result:
                            with st.expander("Raw LLM Response"):
                                st.text(llm_result["raw_response"])
                    else:
                        # Handle case where we have a proper structured result
                        try:
                            # Ensure we have JSON serializable content
                            formatted_json = json.dumps(llm_result, indent=2, default=str)
                            st.code(formatted_json, language="json")
                            
                            # Add download button for the JSON
                            st.download_button(
                                label="Download JSON",
                                data=formatted_json,
                                file_name=f"{uploaded_pdf.name}_receipt_data.json",
                                mime="application/json"
                            )
                            
                            # Show a receipt summary UI
                            with st.expander("Receipt Summary", expanded=True):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    if "store_name" in llm_result:
                                        st.markdown(f"**Store:** {llm_result['store_name']}")
                                    if "date" in llm_result:
                                        st.markdown(f"**Date:** {llm_result['date']}")
                                    if "receipt_no" in llm_result:
                                        st.markdown(f"**Receipt #:** {llm_result['receipt_no']}")
                                
                                with col2:
                                    if "total" in llm_result:
                                        st.markdown(f"**Total:** {llm_result['total']}")
                                    if "subtotal" in llm_result:
                                        st.markdown(f"**Subtotal:** {llm_result['subtotal']}")
                                    if "tax" in llm_result:
                                        st.markdown(f"**Tax:** {llm_result['tax']}")
                                    if "payment_method" in llm_result:
                                        st.markdown(f"**Payment:** {llm_result['payment_method']}")
                                
                                # Display items as a table if they exist
                                if "items" in llm_result and isinstance(llm_result["items"], list) and len(llm_result["items"]) > 0:
                                    st.markdown("### Items")
                                    
                                    # Convert items to DataFrame for better display
                                    items_df = pd.DataFrame(llm_result["items"])
                                    
                                    # Format price column if it exists
                                    if "price" in items_df.columns:
                                        items_df["price"] = items_df["price"].apply(
                                            lambda x: f"${x}" if not str(x).startswith("$") else x
                                        )
                                    
                                    st.dataframe(items_df, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error displaying LLM results: {str(e)}")
                            st.json(llm_result)  # Fallback to raw JSON display
                
                st.download_button(
                    label="Download Raw Text",
                    data=st.session_state.get('pdf_ocr_text', ""),
                    file_name=f"{uploaded_pdf.name}_ocr_text.txt",
                    mime="text/plain"
                )
    
    st.markdown("---")
    st.caption("Receipt OCR App © 2024")

if __name__ == "__main__":
    main()
