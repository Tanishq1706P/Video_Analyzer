import cv2
import torch
import torch.nn.functional as F
import numpy as np
import ultralytics

class ViolenceDetector:
    def __init__(self, yolo_weights=None, device=None, use_slowfast=None):
        """
        Optimized Violence Detector

        Args:
            yolo_weights: YOLO model weights (yolov8n.pt is fastest)
            device: cuda/cpu (auto-detected if None)
            use_slowfast: Enable temporal analysis (can disable for 2x speed)
        """
        from config import Config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_slowfast = use_slowfast if use_slowfast is not None else Config.DETECTORS['violence']['use_slowfast']
        yolo_weights = yolo_weights or Config.DETECTORS['violence']['yolo_weights']

        # Ultralytics YOLO model (already optimized)
        self.yolo = ultralytics.YOLO(yolo_weights)
        
        # SlowFast model (optional, can be slow)
        self.slowfast = None
        if use_slowfast:
            try:
                self.slowfast = torch.hub.load(
                    "facebookresearch/pytorchvideo:main", 
                    "slowfast_r50", 
                    pretrained=True
                ).eval().to(self.device)
            except Exception as e:
                print(f"Warning: Could not load SlowFast model: {e}")
                self.use_slowfast = False

    def _safe_max(self, t):
        if t is None:
            return 0.0
        if isinstance(t, (float, int)):
            return float(t)
        if isinstance(t, torch.Tensor):
            if t.numel() == 0:
                return 0.0
            return float(t.max().cpu().item())
        arr = np.asarray(t)
        if arr.size == 0:
            return 0.0
        return float(arr.max())

    def _calculate_severity(self, confidence):
        if confidence >= 0.75:
            return "high"
        if confidence >= 0.4:
            return "medium"
        if confidence > 0:
            return "low"
        return "none"

    def _process_yolo_batch(self, frames, batch_size=8):
        """Process YOLO detections in batches for speed"""
        yolo_scores = []
        
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            try:
                # Process batch
                results = self.yolo.predict(
                    source=batch, 
                    device=self.device, 
                    verbose=False,
                    half=True if self.device == 'cuda' else False  # FP16 on GPU
                )
                
                for res in results:
                    confs = None
                    if hasattr(res, "boxes"):
                        confs = getattr(res.boxes, "conf", None)
                    yolo_scores.append(self._safe_max(confs))
            except Exception as e:
                print(f"YOLO batch error: {e}")
                yolo_scores.extend([0.0] * len(batch))
        
        return yolo_scores

    def _process_slowfast(self, frames, sample_rate=8, spatial_size=(224, 224), 
                         alpha=4, min_T=4):
        """Optimized SlowFast processing"""
        if not self.use_slowfast or self.slowfast is None:
            return 0.0
        
        # Heavy sampling for speed
        subsampled = frames[::sample_rate]
        
        if len(subsampled) < 2:
            return 0.0
        
        try:
            imgs = []
            for img in subsampled:
                # Convert to tensor
                if isinstance(img, torch.Tensor):
                    t = img.clone()
                else:
                    t = torch.from_numpy(np.asarray(img))

                # Ensure float in 0..1
                if t.dtype == torch.uint8:
                    t = t.float().div(255.0)
                else:
                    t = t.float()

                # Handle H,W,C -> C,H,W
                if t.dim() == 3 and t.shape[-1] in (1, 3, 4):
                    t = t.permute(2, 0, 1)

                if t.dim() == 2:
                    t = t.unsqueeze(0)

                if t.dim() != 3:
                    t = t.view(3, t.shape[-2], t.shape[-1]) if t.numel() >= 3 * t.shape[-2] * t.shape[-1] else t.unsqueeze(0)

                C, H_img, W_img = t.shape

                # Normalize channels to 3
                if C == 1:
                    t = t.repeat(3, 1, 1)
                elif C >= 3:
                    t = t[:3, :, :]
                else:
                    t = torch.cat([t, t[0:1, :, :]], dim=0)

                imgs.append(t.unsqueeze(0))

            # Stack to T x C x H x W
            clip = torch.cat(imgs, dim=0)

            # Spatial resize
            target_h, target_w = spatial_size
            clip = F.interpolate(
                clip, 
                size=(target_h, target_w), 
                mode="bilinear", 
                align_corners=False
            )

            # Convert to (1, C, T, H, W)
            clip = clip.permute(1, 0, 2, 3).unsqueeze(0).to(self.device)

            # Ensure minimum temporal length
            _, C, T, H, W = clip.shape
            if T < min_T:
                pad = clip[:, :, -1:, :, :].repeat(1, 1, (min_T - T), 1, 1)
                clip = torch.cat([clip, pad], dim=2)
                _, C, T, H, W = clip.shape

            # Build pathways
            fast_pathway = clip
            slow_pathway = clip[:, :, ::alpha, :, :]

            # Sanity checks
            if fast_pathway.dim() != 5 or slow_pathway.dim() != 5:
                return 0.0

            # Match channels
            if fast_pathway.size(1) != slow_pathway.size(1):
                cf = fast_pathway.size(1)
                cs = slow_pathway.size(1)
                if cf > cs:
                    fast_pathway = fast_pathway[:, :cs, :, :, :]
                else:
                    slow_pathway = slow_pathway.repeat(1, int(np.ceil(cf / max(1, cs))), 1, 1, 1)[:, :cf, :, :, :]

            # Forward pass
            with torch.no_grad():
                out = self.slowfast([fast_pathway, slow_pathway])
            
            # Extract confidence
            if isinstance(out, dict):
                if "scores" in out:
                    return self._safe_max(out["scores"])
                elif "logits" in out:
                    logits = out["logits"]
                    if isinstance(logits, torch.Tensor):
                        if logits.numel() == 0:
                            return 0.0
                        elif logits.dim() == 2 and logits.size(1) == 1:
                            return float(torch.sigmoid(logits).max().cpu().item())
                        else:
                            return float(torch.softmax(logits, dim=1).max().cpu().item())
                    else:
                        return self._safe_max(logits)
            elif isinstance(out, torch.Tensor):
                if out.numel() == 0:
                    return 0.0
                elif out.dim() == 2 and out.size(1) == 1:
                    return float(torch.sigmoid(out).max().cpu().item())
                elif out.dim() == 2:
                    return float(torch.softmax(out, dim=1).max().cpu().item())
                else:
                    return float(torch.sigmoid(out).max().cpu().item())
            
            return 0.0
            
        except Exception as e:
            print(f"SlowFast error: {e}")
            return 0.0

    def detect(
        self,
        frames,
        slowfast_sample_rate=8,  # Increased from 5 (faster)
        spatial_size=(224, 224),
        alpha=4,
        min_T=4,
        yolo_sample_rate=1,  # New: sample YOLO frames too
        yolo_batch_size=8,   # New: batch YOLO processing
    ):
        """
        Optimized violence detection
        
        Args:
            frames: list of RGB images (H,W,3) as numpy arrays
            slowfast_sample_rate: Higher = faster but less accurate
            yolo_sample_rate: Process every Nth frame with YOLO
            yolo_batch_size: Batch size for YOLO processing
            
        Returns:
            dict with 'category','confidence','severity','components'
        """
        if frames is None or len(frames) == 0:
            return {
                "category": "violence",
                "confidence": 0.0,
                "severity": "none",
                "components": {"weapon": 0.0, "action": 0.0},
            }

        # Sample frames for YOLO (if specified)
        yolo_frames = frames[::yolo_sample_rate]
        
        # YOLO detection with batching
        yolo_scores = self._process_yolo_batch(yolo_frames, yolo_batch_size)
        yolo_conf = float(np.mean(yolo_scores)) if yolo_scores else 0.0

        # Temporal SlowFast (optional)
        temporal_conf = 0.0
        if self.use_slowfast:
            temporal_conf = self._process_slowfast(
                frames, 
                slowfast_sample_rate, 
                spatial_size, 
                alpha, 
                min_T
            )

        # Combine confidences
        weapon_weight = 0.5
        action_weight = 0.5
        confidence = float(weapon_weight * yolo_conf + action_weight * temporal_conf)

        return {
            "category": "violence",
            "confidence": confidence,
            "severity": self._calculate_severity(confidence),
            "components": {
                "weapon": float(yolo_conf), 
                "action": float(temporal_conf)
            },
        }