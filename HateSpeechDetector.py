import easyocr
from transformers import pipeline
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging

class HateSpeechDetector:
    def __init__(self, languages=None, categories=None, use_gpu=None, sample_rate=None,
                 text_density_threshold=None, ocr_scale=None):
        # Use config defaults if not provided
        from config import Config
        self.languages = languages or Config.DETECTORS['hate_speech']['languages']
        self.categories = categories or ["sexual", "education", "funny", "violence"]
        self.use_gpu = use_gpu if use_gpu is not None else Config.DETECTORS['hate_speech']['use_gpu']
        self.sample_rate = sample_rate or Config.DETECTORS['hate_speech']['sample_rate']
        self.text_density_threshold = text_density_threshold or Config.DETECTORS['hate_speech']['text_density_threshold']
        self.ocr_scale = ocr_scale or Config.DETECTORS['hate_speech']['ocr_scale']

        self.reader = easyocr.Reader(self.languages, gpu=self.use_gpu)
        self.classifier = None
    
    def _initialize_classifier(self):
        if self.classifier is None:
            # Use faster model: BART instead of DeBERTa
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if self.use_gpu else -1
            )
    
    def _has_text(self, frame):
        """Quick pre-check if frame has text - improved for text detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Use adaptive thresholding to better detect text
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        # Count non-zero pixels as a proxy for text density
        text_density = np.count_nonzero(thresh) / thresh.size
        return text_density > self.text_density_threshold
    
    def _extract_single(self, frame):
        """Extract text from single frame"""
        try:
            # Resize frame for faster OCR
            small = cv2.resize(frame, None, fx=self.ocr_scale, fy=self.ocr_scale)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            # Enhanced preprocessing for better OCR accuracy
            # Apply bilateral filter to reduce noise while keeping edges
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(filtered)
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)

            result = self.reader.readtext(thresh, paragraph=True)
            return " ".join([r[1] for r in result]).strip()
        except:
            return ""
    
    def extract_text_from_frames(self, frames, sample_rate=None):
        """Extract with smart sampling - reduced sample_rate for better coverage"""
        if sample_rate is None:
            sample_rate = self.sample_rate

        sampled = frames[::sample_rate]

        # Pre-filter frames with text
        text_frames = [f for f in sampled if self._has_text(f)]

        if not text_frames:
            return ""

        # Parallel OCR with reduced workers for speed
        all_text = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            texts = list(executor.map(self._extract_single, text_frames))
            all_text = [t for t in texts if t]

        combined_text = " ".join(all_text)
        logging.getLogger(__name__).info(f"Extracted text: {combined_text[:200]}...")
        return combined_text
    
    def classify_text(self, text):
        if not text or not text.strip():
            return {"labels": [], "scores": [], "text": ""}
        
        # Truncate long text
        if len(text.split()) > 512:
            text = " ".join(text.split()[:512])
        
        self._initialize_classifier()
        result = self.classifier(text, candidate_labels=self.categories)
        result["text"] = text
        return result
    
    def detect(self, frames, sample_rate=None):
        if not frames:
            return {
                "category": "text_analysis",
                "confidence": 0.0,
                "severity": "none",
                "text": "",
                "classifications": []
            }

        text = self.extract_text_from_frames(frames, sample_rate)

        if not text.strip():
            return {
                "category": "text_analysis",
                "confidence": 0.0,
                "severity": "none",
                "text": "No text detected",
                "classifications": []
            }

        classification = self.classify_text(text)

        top_label = classification["labels"][0] if classification["labels"] else "unknown"
        top_score = classification["scores"][0] if classification["scores"] else 0.0

        severity = "none"
        if top_score >= 0.75:
            severity = "high"
        elif top_score >= 0.4:
            severity = "medium"
        elif top_score > 0:
            severity = "low"

        return {
            "category": "text_analysis",
            "confidence": float(top_score),
            "severity": severity,
            "text": text[:200] + "..." if len(text) > 200 else text,
            "top_classification": top_label,
            "all_classifications": [
                {"label": label, "score": float(score)}
                for label, score in zip(classification["labels"], classification["scores"])
            ]
        }
