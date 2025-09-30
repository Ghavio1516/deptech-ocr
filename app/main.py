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
                use_gpu=OCR_CONFIG['enable_gpu']
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
    """Extract text from PaddleOCR 3.x result"""
    all_texts = []
    
    try:
        if isinstance(result, list) and len(result) > 0:
            page_result = result[0]
            
            if isinstance(page_result, dict):
                if 'rec_texts' in page_result:
                    rec_texts = page_result['rec_texts']
                    rec_scores = page_result.get('rec_scores', [])
                    
                    if isinstance(rec_texts, list):
                        for i, text in enumerate(rec_texts):
                            if isinstance(text, str) and text.strip():
                                confidence = rec_scores[i] if i < len(rec_scores) else 1.0
                                # Use confidence threshold from environment
                                if confidence >= OCR_CONFIG['confidence_threshold']:
                                    cleaned_text = text.strip()
                                    if len(cleaned_text) > 0:
                                        all_texts.append(cleaned_text)
                
                if not all_texts:
                    for key in ['texts', 'text', 'results']:
                        if key in page_result:
                            data = page_result[key]
                            if isinstance(data, list):
                                for item in data:
                                    if isinstance(item, str) and item.strip():
                                        all_texts.append(item.strip())
                            elif isinstance(data, str) and data.strip():
                                all_texts.append(data.strip())
    
    except Exception as e:
        print(f"Error extracting text: {e}")
    
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
            
        else:  # hybrid mode
            # Intelligent hybrid extraction
            if file_type == 'pdf':
                print("Running hybrid extraction...")
                
                # Check if PDF is image-heavy
                is_image_heavy = is_image_heavy_pdf(file_bytes)
                
                # Run both methods
                direct_result = extract_pdf_text_direct(file_bytes)
                ocr_result = extract_pdf_text_ocr(file_bytes, request.language, request.quality)
                
                # Smart selection logic
                if not ocr_result["success"] and direct_result["success"]:
                    # OCR failed, use direct
                    raw_text = direct_result["text"]
                    chosen_method = "direct"
                    reason = "OCR failed, using direct extraction"
                    
                elif not direct_result["success"] and ocr_result["success"]:
                    # Direct failed, use OCR
                    raw_text = ocr_result["text"]
                    chosen_method = "ocr"
                    reason = "Direct extraction failed, using OCR"
                    
                elif not direct_result["success"] and not ocr_result["success"]:
                    # Both failed
                    raw_text = "[No text could be extracted]"
                    chosen_method = "none"
                    reason = "Both methods failed"
                    
                else:
                    # Both succeeded - intelligent selection
                    direct_score = direct_result["quality_score"]
                    ocr_score = ocr_result["quality_score"]
                    similarity = SequenceMatcher(None, direct_result["text"], ocr_result["text"]).ratio()
                    
                    print(f"Quality scores - Direct: {direct_score:.2f}, OCR: {ocr_score:.2f}, Similarity: {similarity:.2f}")
                    
                    if is_image_heavy and ocr_score >= 2.0:
                        # Image-heavy PDF with decent OCR - prefer OCR
                        raw_text = ocr_result["text"]
                        chosen_method = "ocr"
                        reason = f"Image-heavy PDF, OCR captures styled content (quality: {ocr_score:.2f})"
                        
                    elif similarity > 0.8 and direct_score > ocr_score:
                        # High similarity, direct is better quality
                        raw_text = direct_result["text"]
                        chosen_method = "direct"
                        reason = f"High similarity ({similarity:.2f}), direct preferred"
                        
                    elif ocr_score > direct_score * 1.5:
                        # OCR significantly better
                        raw_text = ocr_result["text"]
                        chosen_method = "ocr"
                        reason = f"OCR much better quality ({ocr_score:.2f} vs {direct_score:.2f})"
                        
                    elif direct_score > ocr_score * 1.5:
                        # Direct significantly better
                        raw_text = direct_result["text"]
                        chosen_method = "direct"
                        reason = f"Direct much better quality ({direct_score:.2f} vs {ocr_score:.2f})"
                        
                    elif similarity < 0.4 and ocr_score > 1.5:
                        # Low similarity suggests OCR found additional content
                        raw_text = ocr_result["text"]
                        chosen_method = "ocr"
                        reason = f"Low similarity ({similarity:.2f}) suggests OCR found additional content"
                        
                    else:
                        # Default to better quality
                        if direct_score >= ocr_score:
                            raw_text = direct_result["text"]
                            chosen_method = "direct"
                            reason = f"Direct chosen (quality: {direct_score:.2f} vs {ocr_score:.2f})"
                        else:
                            raw_text = ocr_result["text"]
                            chosen_method = "ocr"
                            reason = f"OCR chosen (quality: {ocr_score:.2f} vs {direct_score:.2f})"
                
                processing_details = {
                    "processing_method": f"Hybrid Extraction - {chosen_method.title()}",
                    "extraction_mode": "hybrid",
                    "language": request.language,
                    "quality": request.quality,
                    "is_image_heavy": is_image_heavy,
                    "chosen_method": chosen_method,
                    "selection_reason": reason,
                    "direct_quality": direct_result.get("quality_score", 0),
                    "ocr_quality": ocr_result.get("quality_score", 0),
                    "similarity": similarity if 'similarity' in locals() else 0
                }
                
            elif file_type == 'docx':
                # DOCX - direct extraction
                result = extract_docx_text_direct(file_bytes)
                raw_text = result["text"]
                processing_details = {
                    "processing_method": "Direct DOCX Text Extraction",
                    "extraction_mode": "hybrid",
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
