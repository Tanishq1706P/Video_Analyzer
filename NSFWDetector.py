import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
from PIL import Image

class NSFWDetector:
    def __init__(self, model_path=None, use_pose=None, batch_size=None):
        from config import Config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_pose = use_pose if use_pose is not None else Config.DETECTORS['nsfw']['use_pose']
        self.batch_size = batch_size or Config.DETECTORS['nsfw']['batch_size']
        
        # Use EfficientNet-B0 instead of B7 (8-10x faster)
        self.model = models.efficientnet_b0(pretrained=True)
        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_features, 2)

        # NOTE: This model has random weights and won't detect NSFW properly.
        # For production, replace with a pre-trained NSFW model or train on NSFW dataset.
        # For now, this serves as a placeholder that returns low confidence.
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            except Exception as e:
                print(f"Warning: Could not load NSFW model from {model_path}: {e}")
                print("Using random weights - NSFW detection will be inaccurate.")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Reduced resolution: 224x224 instead of 600x600
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Face blur detector (lightweight)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def predict_batch(self, images):
        """Process multiple images at once"""
        if not images:
            return []
        
        tensors = []
        for image in images:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            tensors.append(self.transform(image))
        
        batch = torch.stack(tensors).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(batch)
            probabilities = torch.softmax(outputs, dim=1)
            nsfw_scores = probabilities[:, 1].cpu().numpy()
        
        return nsfw_scores.tolist()
    
    def detect_blur(self, frame):
        """Fast blur detection"""
        scale = 0.5
        small = cv2.resize(frame, None, fx=scale, fy=scale)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        
        if len(faces) == 0:
            return 0.0
        
        # Scale back
        faces = [(int(x/scale), int(y/scale), int(w/scale), int(h/scale)) 
                 for (x, y, w, h) in faces]
        
        blur_scores = []
        for (x, y, w, h) in faces:
            face_region = frame[y:y+h, x:x+w]
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            blur_scores.append(blur_score)
        
        if blur_scores and np.mean(blur_scores) < 50.0:
            return 0.6
        return 0.0
    
    def detect(self, frames, sample_rate=2):
        """Optimized NSFW detection"""
        if not frames:
            return {
                'category': 'nsfw',
                'confidence': 0.0,
                'severity': 'safe',
                'components': {'nsfw': 0.0, 'blur': 0.0}
            }
        
        # Sample frames
        sampled = frames[::sample_rate]
        
        # Batch process NSFW detection
        nsfw_scores = []
        for i in range(0, len(sampled), self.batch_size):
            batch = sampled[i:i + self.batch_size]
            try:
                batch_scores = self.predict_batch(batch)
                nsfw_scores.extend(batch_scores)
            except:
                nsfw_scores.extend([0.0] * len(batch))
        
        # Quick blur check on sample
        blur_scores = [self.detect_blur(sampled[i]) for i in range(0, len(sampled), 5)]
        
        nsfw_conf = float(np.mean(nsfw_scores)) if nsfw_scores else 0.0
        blur_conf = float(np.mean(blur_scores)) if blur_scores else 0.0
        
        confidence = 0.7 * nsfw_conf + 0.3 * blur_conf
        
        severity = "safe"
        if confidence >= 0.80:
            severity = "critical"
        elif confidence >= 0.60:
            severity = "high"
        elif confidence >= 0.40:
            severity = "medium"
        elif confidence >= 0.20:
            severity = "low"
        
        return {
            'category': 'nsfw',
            'confidence': float(confidence),
            'severity': severity,
            'components': {'nsfw': nsfw_conf, 'blur': blur_conf}
        }