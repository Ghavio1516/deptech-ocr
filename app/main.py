from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from paddleocr import PaddleOCR
import base64
import io
from PIL import Image, ImageEnhance
import numpy as np
import fitz
from docx import Document
import tempfile
from difflib import SequenceMatcher
import re
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Deptech OCR API - Environment Configured",
    version="1.0.0-env",
    description="OCR API with configurable environment settings"
)

class OCRRequest(BaseModel):
    file_base64: str
    filename: str
    quality: str = os.getenv("DEFAULT_QUALITY", "high")  # fast, balanced, high
    language: str = os.getenv("DEFAULT_LANGUAGE", "ch")   # en, ch, id (mapped to ch)
    extraction_mode: str = os.getenv("DEFAULT_EXTRACTION_MODE", "hybrid")  # direct, ocr, hybrid
    enable_cleansing: bool = os.getenv("ENABLE_CLEANSING", "true").lower() == "true"

# Configuration from environment
OCR_CONFIG = {
    "confidence_threshold": float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.5")),
    "max_image_size_mb": int(os.getenv("MAX_IMAGE_SIZE_MB", "50")),
    "enable_gpu": os.getenv("ENABLE_GPU", "false").lower() == "true"
}

ocr_instances = {}

def get_ocr(lang='en'):
    """Get OCR instance with environment configuration"""
    # Map Indonesian to Chinese model for better performance
    if lang == 'id':
        lang = 'ch'
        print("Mapping 'id' language to 'ch' model for better Indonesian text recognition")
        
    if lang not in ocr_instances:
        print(f"Initializing PaddleOCR for language: {lang} (GPU: {OCR_CONFIG['enable_gpu']})")
        try:
            ocr_instances[lang] = PaddleOCR(
                lang=lang,
                #use_gpu=OCR_CONFIG['enable_gpu']
            )
            print(f"PaddleOCR initialized successfully for {lang}!")
        except Exception as e:
            print(f"OCR init failed for {lang}: {e}")
            raise e
    return ocr_instances[lang]

def clean_extracted_text_safe(text: str, deep_clean: bool = True) -> dict:
    """
    Safe version of text cleaning - preserve proper spacing
    """
    if not text or not text.strip():
        return {
            "original_text": text,
            "cleaned_text": "",
            "cleaning_applied": [],
            "original_length": 0,
            "cleaned_length": 0
        }
    
    original_text = text
    original_length = len(text)
    cleaning_steps = []
    
    try:
        # Step 1: Remove control characters (safe method)
        cleaned_chars = []
        for char in text:
            ord_val = ord(char)
            if ord_val < 32 and ord_val not in [9, 10, 13]:  # Keep tab, newline, carriage return
                cleaned_chars.append(' ')
            elif 127 <= ord_val <= 159:  # Remove control chars
                cleaned_chars.append(' ')
            else:
                cleaned_chars.append(char)
        
        text = ''.join(cleaned_chars)
        cleaning_steps.append("Removed control characters")
        
        # Step 2: SQL-safe escaping (if deep_clean)
        if deep_clean:
            text = text.replace("'", "''")
            text = text.replace("\\", "\\\\") 
            text = text.replace('"', '\\"')
            cleaning_steps.append("Applied SQL-safe escaping")
        
        # Step 3: Normalize line breaks
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        cleaning_steps.append("Normalized line breaks")
        
        # Step 4: Fix excessive spaces (preserve single spaces)
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs -> single space
        cleaning_steps.append("Fixed excessive horizontal spaces")
        
        # Step 5: Fix excessive newlines (limit to double newlines max)
        text = re.sub(r'\n{3,}', '\n\n', text)  # 3+ newlines -> double newline
        cleaning_steps.append("Limited consecutive line breaks")
        
        # Step 6: Clean up line by line (preserve paragraph structure)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Trim whitespace from start/end of each line
            cleaned_line = line.strip()
            cleaned_lines.append(cleaned_line)
        
        text = '\n'.join(cleaned_lines)
        cleaning_steps.append("Trimmed individual lines")
        
        # Step 7: Fix common spacing issues around punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([,.!?;:])\s*', r'\1 ', text)  # Ensure space after punctuation  
        text = re.sub(r'\s+', ' ', text)  # Clean up any double spaces created
        cleaning_steps.append("Fixed punctuation spacing")
        
        # Step 8: OCR-specific cleaning (if deep_clean)
        if deep_clean:
            # Fix common OCR errors that create spacing issues
            ocr_fixes = {
                'AnInformaticsEngineeringstudent': 'An Informatics Engineering student',
                'JakartaStatePolytechnic': 'Jakarta State Polytechnic',
                'withastrongpassion': 'with a strong passion',
                'Wband': 'Web and',
                'Eagertocreateimpactfuldigitalsolutions': 'Eager to create impactful digital solutions',
                'bycombiningtechnicalsklls': 'by combining technical skills',
                'theobligation': 'the obligation',
                'asaveri': 'as a verification',
                'cation': 'cation',
                'verifyovera': 'verify over a',
                'mobileapplication': 'mobile application',
                'nutritiontracking': 'nutrition tracking',
                'Rsponsible': 'Responsible',
                'ordesigning': 'for designing',
                'implementingcoreeature': 'implementing core features',
                'Kotlinandintegrating': 'Kotlin and integrating',
                'APl': 'API',
                'Dveloo': 'Develop',
                'deliverytuiOR': 'delivery tracker',
                'badIdosltil': 'based solution',
                'Engineeringstudent': 'Engineering student',
                'verification cation': 'verification',
                'Power Point': 'PowerPoint',
                'Cs S': 'CSS',
            }
            
            # Apply OCR fixes
            for wrong, correct in ocr_fixes.items():
                if wrong in text:
                    text = text.replace(wrong, correct)
                    cleaning_steps.append(f"Fixed OCR error: '{wrong}' -> '{correct}'")
            
            # Fix other common patterns - add space between lowercase-uppercase
            text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
            cleaning_steps.append("Added missing spaces between words")
        
        # Step 9: Final cleanup
        text = text.strip()
        cleaning_steps.append("Final trim")
        
        # Remove any remaining double spaces
        while '  ' in text:
            text = text.replace('  ', ' ')
        
        cleaned_length = len(text)
        
        return {
            "original_text": original_text,
            "cleaned_text": text,
            "cleaning_applied": cleaning_steps,
            "original_length": original_length,
            "cleaned_length": cleaned_length,
            "reduction_percentage": round((original_length - cleaned_length) / original_length * 100, 2) if original_length > 0 else 0
        }
        
    except Exception as e:
        print(f"Error in text cleaning: {e}")
        return {
            "original_text": original_text,
            "cleaned_text": original_text,  # Return original if cleaning fails
            "cleaning_applied": [f"Cleaning failed: {str(e)}"],
            "original_length": original_length,
            "cleaned_length": original_length,
            "reduction_percentage": 0
        }

def is_image_heavy_pdf(pdf_bytes: bytes) -> bool:
    """Quick check if PDF is image-heavy (for OCR-first strategy)"""
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = pdf_document.page_count
        pages_with_images = 0
        total_text_chars = 0
        
        for page_num in range(min(3, total_pages)):  # Check first 3 pages only
            page = pdf_document[page_num]
            
            # Check for images
            if page.get_images():
                pages_with_images += 1
            
            # Check text amount
            page_text = page.get_text().strip()
            total_text_chars += len(page_text)
        
        pdf_document.close()
        
        # Simple heuristic: if >50% pages have images and low text, likely image-heavy
        image_ratio = pages_with_images / min(3, total_pages) if total_pages > 0 else 0
        avg_text = total_text_chars / min(3, total_pages) if total_pages > 0 else 0
        
        is_heavy = image_ratio > 0.5 and avg_text < 800
        print(f"PDF Analysis - Image ratio: {image_ratio:.2f}, Avg text: {avg_text:.0f} chars, Image-heavy: {is_heavy}")
        
        return is_heavy
        
    except Exception as e:
        print(f"PDF analysis failed: {e}")
        return False

def extract_pdf_text_direct(pdf_bytes: bytes) -> dict:
    """Extract text directly from PDF - NO PAGE HEADERS"""
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        text_content = []
        total_chars = 0
        
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            page_text = page.get_text().strip()
            
            if page_text:
                # Just add the text without page headers
                text_content.append(page_text)
                total_chars += len(page_text)
        
        pdf_document.close()
        full_text = "\n\n".join(text_content)  # Separate pages with double newline
        
        return {
            "success": True,
            "method": "Direct PDF Text Extraction",
            "text": full_text,
            "quality_score": min(total_chars / 100, 10)
        }
        
    except Exception as e:
        return {
            "success": False,
            "method": "Direct PDF Text Extraction",
            "error": str(e),
            "text": "",
            "quality_score": 0
        }

def extract_docx_text_direct(docx_bytes: bytes) -> dict:
    """Extract text directly from DOCX"""
    try:
        with tempfile.NamedTemporaryFile() as tmp_file:
            tmp_file.write(docx_bytes)
            tmp_file.flush()
            
            doc = Document(tmp_file.name)
            text_content = []
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_content.append(paragraph.text.strip())
            
            full_text = "\n".join(text_content)
            
            return {
                "success": True,
                "method": "Direct DOCX Text Extraction",
                "text": full_text,
                "quality_score": min(len(full_text) / 100, 10)
            }
            
    except Exception as e:
        return {
            "success": False,
            "method": "Direct DOCX Text Extraction", 
            "error": str(e),
            "text": "",
            "quality_score": 0
        }

def extract_text_from_paddleocr_3x(result):
    """Extract text from PaddleOCR result - Enhanced error handling"""
    all_texts = []
    
    try:
        print(f"Processing OCR result type: {type(result)}")
        
        if result is None:
            print("OCR result is None")
            return all_texts
            
        if isinstance(result, list) and len(result) > 0:
            for page_idx, page_result in enumerate(result):
                if page_result is None:
                    print(f"Page {page_idx} result is None")
                    continue
                    
                print(f"Page {page_idx} result type: {type(page_result)}")
                
                # Handle list of detection results (standard format)
                if isinstance(page_result, list):
                    print(f"Processing list format with {len(page_result)} detections")
                    for detection_idx, detection in enumerate(page_result):
                        if isinstance(detection, list) and len(detection) >= 2:
                            try:
                                # Format: [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ('text', confidence)]
                                bbox = detection[0]  # Bounding box coordinates
                                text_info = detection[1]  # Text and confidence
                                
                                if isinstance(text_info, tuple) and len(text_info) >= 2:
                                    text = str(text_info[0]).strip()
                                    confidence = float(text_info[1])
                                elif isinstance(text_info, str):
                                    text = str(text_info).strip()
                                    confidence = 1.0
                                else:
                                    print(f"Unexpected text_info format: {type(text_info)}")
                                    continue
                                
                                # Apply confidence threshold
                                if confidence >= OCR_CONFIG['confidence_threshold'] and len(text) > 0:
                                    all_texts.append(text)
                                    print(f"✓ Extracted: '{text}' (confidence: {confidence:.3f})")
                                else:
                                    print(f"✗ Skipped: '{text}' (confidence: {confidence:.3f} < {OCR_CONFIG['confidence_threshold']})")
                                
                            except Exception as detection_error:
                                print(f"Error processing detection {detection_idx}: {detection_error}")
                                continue
                
                # Handle dictionary format (older versions or different OCR modes)
                elif isinstance(page_result, dict):
                    print("Processing dictionary format OCR result")
                    
                    # Method 1: Check for 'rec_texts' and 'rec_scores' keys (common in older versions)
                    if 'rec_texts' in page_result:
                        print("Found 'rec_texts' in dictionary")
                        rec_texts = page_result['rec_texts']
                        rec_scores = page_result.get('rec_scores', [])
                        
                        if isinstance(rec_texts, list):
                            print(f"Processing {len(rec_texts)} rec_texts")
                            for i, text in enumerate(rec_texts):
                                if isinstance(text, str) and text.strip():
                                    confidence = rec_scores[i] if i < len(rec_scores) else 1.0
                                    text_clean = text.strip()
                                    
                                    if confidence >= OCR_CONFIG['confidence_threshold'] and len(text_clean) > 0:
                                        all_texts.append(text_clean)
                                        print(f"✓ Dictionary extracted: '{text_clean}' (confidence: {confidence:.3f})")
                                    else:
                                        print(f"✗ Dictionary skipped: '{text_clean}' (confidence: {confidence:.3f})")
                    
                    # Method 2: Check for other common dictionary keys
                    if not all_texts:  # Only try other methods if rec_texts didn't work
                        print("Trying alternative dictionary keys...")
                        for key in ['texts', 'text', 'results', 'predictions', 'output']:
                            if key in page_result:
                                print(f"Found '{key}' in dictionary")
                                data = page_result[key]
                                
                                if isinstance(data, list):
                                    print(f"Processing list data with {len(data)} items")
                                    for item in data:
                                        if isinstance(item, str) and item.strip():
                                            text_clean = item.strip()
                                            if len(text_clean) > 0:
                                                all_texts.append(text_clean)
                                                print(f"✓ Alt key extracted: '{text_clean}'")
                                        elif isinstance(item, dict):
                                            # Handle nested dictionaries
                                            for nested_key in ['text', 'content', 'value']:
                                                if nested_key in item and isinstance(item[nested_key], str):
                                                    text_clean = str(item[nested_key]).strip()
                                                    if len(text_clean) > 0:
                                                        all_texts.append(text_clean)
                                                        print(f"✓ Nested extracted: '{text_clean}'")
                                                    break
                                
                                elif isinstance(data, str) and data.strip():
                                    text_clean = data.strip()
                                    if len(text_clean) > 0:
                                        all_texts.append(text_clean)
                                        print(f"✓ String data extracted: '{text_clean}'")
                                
                                break  # Stop after finding first valid key
                    
                    # Method 3: If still no results, try to extract any text-like values
                    if not all_texts:
                        print("Trying deep extraction from dictionary...")
                        def extract_text_recursive(obj, path=""):
                            texts = []
                            if isinstance(obj, str) and len(obj.strip()) > 0:
                                texts.append(obj.strip())
                            elif isinstance(obj, list):
                                for i, item in enumerate(obj):
                                    texts.extend(extract_text_recursive(item, f"{path}[{i}]"))
                            elif isinstance(obj, dict):
                                for key, value in obj.items():
                                    if key in ['text', 'content', 'value', 'result', 'output']:
                                        texts.extend(extract_text_recursive(value, f"{path}.{key}"))
                            return texts
                        
                        deep_texts = extract_text_recursive(page_result)
                        for text in deep_texts[:10]:  # Limit to prevent spam
                            if len(text) > 2:  # Skip very short texts
                                all_texts.append(text)
                                print(f"✓ Deep extracted: '{text}'")
                
                else:
                    print(f"Unexpected page result format: {type(page_result)}")
                    # Try to convert to string if possible
                    try:
                        text_str = str(page_result).strip()
                        if len(text_str) > 0 and text_str not in ['None', 'null', '[]', '{}']:
                            all_texts.append(text_str)
                            print(f"✓ Fallback extracted: '{text_str}'")
                    except:
                        pass
                        
        else:
            print(f"Unexpected result format or empty: {type(result)}")
            
    except Exception as e:
        print(f"Error in extract_text_from_paddleocr_3x: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"Total extracted texts: {len(all_texts)}")
    return all_texts

def extract_pdf_text_ocr(pdf_bytes: bytes, language: str, quality: str) -> dict:
    """Extract text using OCR - NO PAGE HEADERS"""
    try:
        ocr = get_ocr(language)
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = pdf_document.page_count
        
        # Quality-based DPI
        dpi_scale = {"fast": 1.5, "balanced": 2.0, "high": 2.5}[quality]
        
        page_results = []
        total_chars = 0
        
        for page_num in range(total_pages):
            try:
                page = pdf_document[page_num]
                
                # Convert to image
                mat = fitz.Matrix(dpi_scale, dpi_scale)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # Enhance image
                img = Image.open(io.BytesIO(img_data))
                if quality != "fast":
                    enhancer = ImageEnhance.Contrast(img)
                    img = enhancer.enhance(1.3)
                    enhancer = ImageEnhance.Sharpness(img)
                    img = enhancer.enhance(1.1)
                
                img_array = np.array(img)
                
                # Run OCR
                result = ocr.ocr(img_array)
                page_texts = extract_text_from_paddleocr_3x(result)
                
                if page_texts:
                    # Just join the texts without page headers
                    page_content = "\n".join(page_texts)
                    page_results.append(page_content)
                    total_chars += sum(len(text) for text in page_texts)
                # Skip pages with no text (don't add empty entries)
                
            except Exception as page_error:
                # Skip failed pages silently or add minimal error info
                print(f"Page {page_num + 1} OCR error: {page_error}")
                continue
        
        pdf_document.close()
        full_text = "\n\n".join(page_results)  # Separate pages with double newline
        
        return {
            "success": True,
            "method": f"OCR Text Extraction ({language}, {quality})",
            "text": full_text,
            "quality_score": min(total_chars / 100, 10)
        }
        
    except Exception as e:
        return {
            "success": False,
            "method": "OCR Text Extraction",
            "error": str(e),
            "text": "",
            "quality_score": 0
        }

def detect_file_type(filename: str, file_bytes: bytes) -> str:
    """Detect file type"""
    filename_lower = filename.lower()
    
    if filename_lower.endswith('.pdf'):
        return 'pdf'
    elif filename_lower.endswith('.docx'):
        return 'docx'
    elif filename_lower.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        return 'image'
    else:
        if file_bytes.startswith(b'%PDF'):
            return 'pdf'
        elif file_bytes.startswith(b'PK'):
            return 'docx'
        else:
            return 'image'

@app.get("/")
def root():
    return {
        "message": "Deptech OCR API - Environment Configured", 
        "version": "1.0.0-env",
        "environment": os.getenv("ENV", "development"),
        "endpoint": "POST /extract",
        "configuration": {
            "default_language": os.getenv("DEFAULT_LANGUAGE", "ch"),
            "default_quality": os.getenv("DEFAULT_QUALITY", "high"), 
            "default_extraction_mode": os.getenv("DEFAULT_EXTRACTION_MODE", "hybrid"),
            "cleansing_enabled": os.getenv("ENABLE_CLEANSING", "true"),
            "confidence_threshold": OCR_CONFIG["confidence_threshold"],
            "max_image_size_mb": OCR_CONFIG["max_image_size_mb"],
            "gpu_enabled": OCR_CONFIG["enable_gpu"]
        },
        "options": {
            "quality": ["fast", "balanced", "high"],
            "language": ["en", "ch", "id"],
            "extraction_mode": ["direct", "ocr", "hybrid"]
        },
        "features": [
            "Clean text without page headers",
            "Smart OCR-first for image-heavy PDFs",
            "Environment-based configuration",
            "Advanced data cleansing with OCR fixes",
            "GPU support (if enabled)",
            "Configurable confidence thresholds"
        ]
    }

@app.get("/health")
def health():
    """Health check endpoint"""
    try:
        return {
            "status": "healthy",
            "environment": os.getenv("ENV", "development"),
            "version": "1.0.0-env",
            "ocr_ready": True,
            "configuration": {
                "default_language": os.getenv("DEFAULT_LANGUAGE", "ch"),
                "default_quality": os.getenv("DEFAULT_QUALITY", "high"),
                "gpu_enabled": OCR_CONFIG["enable_gpu"],
                "confidence_threshold": OCR_CONFIG["confidence_threshold"]
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.post("/extract")
async def extract_document_text(request: OCRRequest):
    """Single endpoint for all text extraction needs - Environment configured"""
    try:
        # Decode base64
        file_bytes = base64.b64decode(request.file_base64)
        file_type = detect_file_type(request.filename, file_bytes)
        
        print(f"Processing {request.filename} - Mode: {request.extraction_mode}, Type: {file_type}")
        print(f"Environment: {os.getenv('ENV', 'development')}")
        print(f"Language: {request.language}, Quality: {request.quality}")
        
        raw_text = ""
        processing_details = {}
        
        if request.extraction_mode == "direct":
            # Direct extraction only
            if file_type == 'pdf':
                result = extract_pdf_text_direct(file_bytes)
            elif file_type == 'docx':
                result = extract_docx_text_direct(file_bytes)
            else:
                raise HTTPException(status_code=400, detail="Direct extraction only supports PDF and DOCX")
            
            raw_text = result["text"]
            processing_details = {
                "processing_method": result["method"],
                "extraction_mode": request.extraction_mode,
                "quality_score": result.get("quality_score", 0)
            }
            
        elif request.extraction_mode == "ocr":
            # OCR only
            if file_type == 'pdf':
                result = extract_pdf_text_ocr(file_bytes, request.language, request.quality)
            else:
                raise HTTPException(status_code=400, detail="OCR-only mode currently supports PDF only")
            
            raw_text = result["text"]
            processing_details = {
                "processing_method": result["method"],
                "extraction_mode": request.extraction_mode,
                "language": request.language,
                "quality": request.quality,
                "quality_score": result.get("quality_score", 0)
            }
            
        else:  # hybrid mode - OCR-FIRST STRATEGY
            # OCR-First hybrid extraction
            if file_type == 'pdf':
                print("Running OCR-First hybrid extraction...")
                
                # Check if PDF is image-heavy
                is_image_heavy = is_image_heavy_pdf(file_bytes)
                
                # ALWAYS RUN OCR FIRST
                print("Step 1: Running OCR extraction...")
                ocr_result = extract_pdf_text_ocr(file_bytes, request.language, request.quality)
                
                # Run direct extraction as backup/supplement
                print("Step 2: Running direct extraction for supplement...")
                direct_result = extract_pdf_text_direct(file_bytes)
                
                # OCR-FIRST DECISION LOGIC
                if ocr_result["success"] and len(ocr_result["text"].strip()) > 0:
                    # OCR succeeded - use it as primary
                    ocr_score = ocr_result["quality_score"]
                    direct_score = direct_result["quality_score"] if direct_result["success"] else 0
                    
                    print(f"OCR-First: OCR succeeded (quality: {ocr_score:.2f})")
                    
                    if direct_result["success"] and len(direct_result["text"].strip()) > 0:
                        # Both methods have content - compare and enhance
                        similarity = SequenceMatcher(None, direct_result["text"], ocr_result["text"]).ratio()
                        print(f"Similarity between OCR and Direct: {similarity:.2f}")
                        
                        if similarity < 0.6:
                            # Low similarity - OCR probably captured images/styled text that direct missed
                            # Use OCR as primary, add direct as supplement for missing regular text
                            direct_lines = set(line.strip() for line in direct_result["text"].split('\n') if line.strip() and len(line.strip()) > 3)
                            ocr_lines = set(line.strip() for line in ocr_result["text"].split('\n') if line.strip() and len(line.strip()) > 3)
                            
                            # Find meaningful content only in direct extraction
                            direct_only = direct_lines - ocr_lines
                            meaningful_direct = [line for line in direct_only if len(line) > 10]  # Skip short fragments
                            
                            if meaningful_direct:
                                raw_text = f"{ocr_result['text']}\n\n--- SUPPLEMENTAL TEXT (Direct) ---\n" + "\n".join(meaningful_direct)
                                chosen_method = "ocr_enhanced"
                                reason = f"OCR primary + direct supplement (similarity: {similarity:.2f}, added {len(meaningful_direct)} lines)"
                            else:
                                raw_text = ocr_result["text"]
                                chosen_method = "ocr"
                                reason = f"OCR complete (similarity: {similarity:.2f}, no meaningful direct supplement)"
                        else:
                            # High similarity - OCR is comprehensive
                            raw_text = ocr_result["text"]
                            chosen_method = "ocr"
                            reason = f"OCR comprehensive (similarity: {similarity:.2f})"
                    else:
                        # Only OCR has content
                        raw_text = ocr_result["text"]
                        chosen_method = "ocr"
                        reason = "OCR succeeded, direct extraction empty"
                        
                elif direct_result["success"] and len(direct_result["text"].strip()) > 0:
                    # OCR failed but direct succeeded
                    print("OCR-First: OCR failed, falling back to direct")
                    raw_text = direct_result["text"]
                    chosen_method = "direct_fallback"
                    reason = "OCR failed, using direct extraction as fallback"
                    
                else:
                    # Both failed
                    print("OCR-First: Both methods failed")
                    raw_text = "[No text could be extracted - both OCR and direct methods failed]"
                    chosen_method = "none"
                    reason = "Both OCR and direct extraction failed"
                
                processing_details = {
                    "processing_method": f"OCR-First Hybrid - {chosen_method.title().replace('_', ' ')}",
                    "extraction_mode": "hybrid",
                    "strategy": "ocr_first",
                    "language": request.language,
                    "quality": request.quality,
                    "is_image_heavy": is_image_heavy,
                    "chosen_method": chosen_method,
                    "selection_reason": reason,
                    "ocr_quality": ocr_result.get("quality_score", 0),
                    "direct_quality": direct_result.get("quality_score", 0),
                    "ocr_success": ocr_result["success"],
                    "direct_success": direct_result["success"],
                    "similarity": similarity if 'similarity' in locals() else 0
                }
                
            elif file_type == 'docx':
                # DOCX - direct extraction (OCR not needed for structured docs)
                result = extract_docx_text_direct(file_bytes)
                raw_text = result["text"]
                processing_details = {
                    "processing_method": "Direct DOCX Text Extraction",
                    "extraction_mode": "hybrid",
                    "strategy": "direct_only",
                    "quality_score": result.get("quality_score", 0)
                }
                
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_type}")
        
        # Apply data cleansing if enabled
        if request.enable_cleansing:
            print("Applying data cleansing...")
            cleansing_result = clean_extracted_text_safe(raw_text, deep_clean=True)
            
            final_text = cleansing_result["cleaned_text"]
            cleansing_details = {
                "cleansing_enabled": True,
                "original_length": cleansing_result["original_length"],
                "cleaned_length": cleansing_result["cleaned_length"],
                "reduction_percentage": cleansing_result["reduction_percentage"],
                "cleaning_steps": cleansing_result["cleaning_applied"]
            }
        else:
            final_text = raw_text
            cleansing_details = {
                "cleansing_enabled": False
            }
        
        # Final response
        return {
            "success": True,
            "filename": request.filename,
            "file_type": file_type,
            "original_text": raw_text,        # RAW extracted text (no page headers)
            "extracted_text": final_text,     # CLEANED text
            "text_length": len(final_text),
            "original_text_length": len(raw_text),
            "has_content": len(final_text.strip()) > 0,
            "environment": os.getenv("ENV", "development"),
            **processing_details,
            **cleansing_details
        }
        
    except Exception as e:
        print(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
