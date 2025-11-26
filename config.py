# Configuration file for video analyzer
import os

class Config:
    # Video processing
    VIDEO_PATH = "Sample_video.mp4"
    FRAME_RATE = 5  # Extract 1 frame every N seconds
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480

    # Detector configurations
    DETECTORS = {
        'violence': {
            'use_slowfast': True,  # Disable for speed
            'yolo_weights': 'yolov8n.pt'
        },
        'nsfw': {
            'batch_size': 16,
            'use_pose': True
        },
        'hate_speech': {
            'sample_rate': 2,
            'text_density_threshold': 0.05,
            'ocr_scale': 0.75,
            'languages': ['en'],
            'use_gpu': False
        },
        'audio': {
            'whisper_model': 'tiny',
            'use_gpu': False
        }
    }

    # Output
    OUTPUT_FOLDER = "_frames"
    LOG_LEVEL = "INFO"

    # Performance
    MAX_WORKERS = 4
    CLEANUP_TEMP_FILES = True
