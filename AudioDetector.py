import os
import subprocess
import torch
import whisper
from transformers import pipeline

class AudioDetector:
    def __init__(self, whisper_model=None, categories=None, use_gpu=None):
        # Use 'tiny' model by default (much faster than 'base')
        from config import Config
        self.whisper_model_name = whisper_model or Config.DETECTORS['audio']['whisper_model']
        self.whisper_model = None
        self.categories = categories or ["sexual", "education", "funny", "violence"]
        self.classifier = None
        self.use_gpu = use_gpu if use_gpu is not None else Config.DETECTORS['audio']['use_gpu']
    
    def _initialize_whisper(self):
        if self.whisper_model is None:
            self.whisper_model = whisper.load_model(self.whisper_model_name)
    
    def _initialize_classifier(self):
        if self.classifier is None:
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if self.use_gpu else -1
            )
    
    def check_audio_exists(self, video_path):
        command = [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=codec_type",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path
        ]
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE, text=True, timeout=5)
            return "audio" in result.stdout.lower()
        except:
            return False
    
    def extract_audio(self, video_path, output_path="temp_audio.wav"):
        if not self.check_audio_exists(video_path):
            return None
        
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        command = [
            "ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1", output_path, "-y"  # Mono for faster processing
        ]
        
        result = subprocess.run(command, stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE, timeout=30)
        
        if result.returncode != 0:
            return None
        return output_path
    
    def transcribe_audio(self, audio_path, language=None):
        self._initialize_whisper()
        # Use fp16 for speed if on GPU
        result = self.whisper_model.transcribe(
            audio_path, 
            language=language,
            fp16=torch.cuda.is_available()
        )
        return result["text"]
    
    def classify_text(self, text):
        if not text or not text.strip():
            return {"labels": [], "scores": [], "text": ""}
        
        if len(text.split()) > 512:
            text = " ".join(text.split()[:512])
        
        self._initialize_classifier()
        result = self.classifier(text, candidate_labels=self.categories)
        result["text"] = text
        return result
    
    def detect(self, video_path, language=None, cleanup=True):
        audio_path = None
        try:
            audio_path = self.extract_audio(video_path)
            
            if audio_path is None:
                return {
                    "category": "audio_analysis",
                    "confidence": 0.0,
                    "severity": "none",
                    "transcription": "",
                    "classifications": [],
                    "has_audio": False
                }
            
            text = self.transcribe_audio(audio_path, language)
            
            if not text.strip():
                return {
                    "category": "audio_analysis",
                    "confidence": 0.0,
                    "severity": "none",
                    "transcription": "",
                    "classifications": [],
                    "has_audio": True
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
                "category": "audio_analysis",
                "confidence": float(top_score),
                "severity": severity,
                "transcription": text,
                "top_classification": top_label,
                "all_classifications": [
                    {"label": label, "score": float(score)}
                    for label, score in zip(classification["labels"], classification["scores"])
                ],
                "has_audio": True
            }
        finally:
            if cleanup and audio_path and os.path.exists(audio_path):
                os.remove(audio_path)