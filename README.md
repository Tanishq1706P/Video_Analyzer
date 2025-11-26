# Video Content Analyzer

A production-grade video analysis system that detects violence, NSFW content, hate speech, and analyzes audio using advanced AI models.

## Features

- **Multi-Detector Analysis**: Simultaneous detection of violence, NSFW content, hate speech, and audio analysis
- **Optimized Performance**: Configurable frame rates, batch processing, and parallel execution
- **Production Ready**: Logging, error handling, configuration management, and structured output
- **Scalable Architecture**: Modular detector classes with configurable parameters

## Installation

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.py` to customize:

- Video processing parameters (frame rate, resolution)
- Detector-specific settings
- Performance tuning options

## Usage

```bash
python main.py
```

The system will:

1. Extract frames from the configured video
2. Run all detectors in parallel
3. Generate a comprehensive analysis report
4. Save results to `analysis_results.json`

## Detectors

### ViolenceDetector

- Uses YOLOv8 for weapon detection
- Optional SlowFast for temporal analysis
- Configurable confidence thresholds

### NSFWDetector

- EfficientNet-based classification
- Face blur detection
- Batch processing for speed

### HateSpeechDetector

- EasyOCR for text extraction
- Advanced image preprocessing (CLAHE, bilateral filtering)
- Zero-shot classification with BART

### AudioDetector

- Whisper for transcription
- Zero-shot audio content classification
- Optimized for speed with tiny model

## Output

Results include:

- Confidence scores for each category
- Severity levels (none/low/medium/high/critical)
- Component breakdowns
- Extracted text/audio transcriptions
- Processing times

## Performance Optimization

- Frame resizing and sampling
- Parallel processing with ThreadPoolExecutor
- GPU acceleration where available
- Configurable batch sizes and workers

## Architecture

- **config.py**: Centralized configuration
- **main.py**: Orchestration and result handling
- **{Detector}.py**: Specialized detection modules
- Modular design for easy extension

## Requirements

- Python 3.8+
- FFmpeg (for audio processing)
- CUDA-compatible GPU (optional, for acceleration)
