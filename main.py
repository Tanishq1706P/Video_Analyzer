# ===== Production-Grade Video Analyzer =====
import os
import cv2
import shutil
import time
import logging
import argparse
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from ViolenceDetector import ViolenceDetector
from NSFWDetector import NSFWDetector
from HateSpeechDetector import HateSpeechDetector
from AudioDetector import AudioDetector
from config import Config

# Suppress all library warnings to console
warnings.filterwarnings('ignore')

def setup_logging(log_file="video_analyzer.log"):
    """Setup logging configuration with file output only (no console warnings)"""
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, Config.LOG_LEVEL))

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File handler only (no console output)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def extract_frames(video_path, output_folder=None, frame_rate=None):
    """Extract frames with optimized processing and memory efficiency"""
    if output_folder is None:
        output_folder = Config.OUTPUT_FOLDER
    if frame_rate is None:
        frame_rate = Config.FRAME_RATE

    os.makedirs(output_folder, exist_ok=True)
    logger = logging.getLogger(__name__)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    logger.info(f"Video: {total_frames} frames, {fps:.1f} fps, {duration:.1f}s duration")
    logger.info(f"Extracting 1 frame every {frame_rate} seconds...")

    interval = int(fps * frame_rate)
    frame_count = 0
    saved_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % interval == 0:
                # Resize frame for consistent processing
                frame = cv2.resize(frame, (Config.FRAME_WIDTH, Config.FRAME_HEIGHT))
                frame_filename = os.path.join(output_folder, f"frame_{saved_count:06d}.jpg")
                cv2.imwrite(frame_filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                saved_count += 1

            frame_count += 1
    finally:
        cap.release()

    logger.info(f"Extracted {saved_count} frames")
    return saved_count  # Return count instead of loading all frames into memory


def run_detector(detector_name, detector, data):
    """Generic detector runner with timing and error handling"""
    logger = logging.getLogger(__name__)
    start = time.time()
    try:
        result = detector.detect(data)
        elapsed = time.time() - start
        logger.info(f"✓ {detector_name} completed in {elapsed:.2f}s")
        return (detector_name, result, elapsed)
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"✗ {detector_name} failed: {e}")
        return (detector_name, {"error": str(e)}, elapsed)

def run_detector_batch(detector_name, detector, output_folder):
    """Run detector with batch processing for memory efficiency"""
    logger = logging.getLogger(__name__)
    start = time.time()
    try:
        all_frames = []
        for batch in load_frames_from_folder(output_folder):
            all_frames.extend(batch)
        result = detector.detect(all_frames)
        elapsed = time.time() - start
        logger.info(f"✓ {detector_name} completed in {elapsed:.2f}s")
        return (detector_name, result, elapsed)
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"✗ {detector_name} failed: {e}")
        return (detector_name, {"error": str(e)}, elapsed)


def load_frames_from_folder(output_folder, batch_size=50):
    """Load frames from folder in batches to save memory"""
    frame_files = sorted([f for f in os.listdir(output_folder) if f.endswith('.jpg')])
    for i in range(0, len(frame_files), batch_size):
        batch_files = frame_files[i:i + batch_size]
        frames = []
        for f in batch_files:
            frame_path = os.path.join(output_folder, f)
            frame = cv2.imread(frame_path)
            if frame is not None:
                frames.append(frame)
        yield frames

def analyze_video_with_load_balancing(video_path, output_folder=None):
    """
    Production-grade video analysis with intelligent load balancing and memory optimization
    """
    logger = setup_logging()

    logger.info("="*70)
    logger.info("PRODUCTION-GRADE VIDEO ANALYSIS")
    logger.info("="*70)

    overall_start = time.time()

    # Extract frames with config settings
    logger.info("\n[1/3] Extracting frames...")
    frame_count = extract_frames(video_path, output_folder)

    if frame_count == 0:
        logger.error("No frames extracted")
        return {}

    # Load frames only when needed for detectors
    logger.info(f"Extracted {frame_count} frames to disk")

    logger.info(f"\n[2/3] Loading frames for analysis...")

    # Check if frames exist
    output_dir = output_folder or Config.OUTPUT_FOLDER
    if not os.path.exists(output_dir) or not os.listdir(output_dir):
        logger.error("No frames found in output folder")
        return {}

    logger.info(f"\n[3/3] Running detectors in parallel...")
    logger.info("-"*70)

    results = {}

    # Initialize detectors with config
    detectors = {
        'violence': ViolenceDetector(**Config.DETECTORS['violence']),
        'nsfw': NSFWDetector(**Config.DETECTORS['nsfw']),
        'hate_speech': HateSpeechDetector(**Config.DETECTORS['hate_speech']),
        'audio': AudioDetector(**Config.DETECTORS['audio'])
    }

    # Use ThreadPoolExecutor with configurable workers
    max_workers = min(Config.MAX_WORKERS, len(detectors))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        # Submit frame-based detectors with batch processing
        for name in ['violence', 'nsfw', 'hate_speech']:
            future = executor.submit(run_detector_batch, name, detectors[name], output_dir)
            futures[future] = name

        # Submit audio detector
        future = executor.submit(run_detector, 'audio', detectors['audio'], video_path)
        futures[future] = 'audio'

        # Collect results as they complete
        for future in as_completed(futures):
            name, result, elapsed = future.result()
            results[name] = result

    # Cleanup temp files
    if Config.CLEANUP_TEMP_FILES and os.path.exists(output_folder or Config.OUTPUT_FOLDER):
        shutil.rmtree(output_folder or Config.OUTPUT_FOLDER, ignore_errors=True)

    overall_elapsed = time.time() - overall_start

    logger.info("-"*70)
    logger.info(f"\n✓ Total time: {overall_elapsed:.2f}s")
    logger.info("="*70)

    return results


def print_results(results):
    """Pretty print results with structured output"""
    logger = logging.getLogger(__name__)

    logger.info("\n" + "="*70)
    logger.info("DETECTION RESULTS")
    logger.info("="*70)

    for name, result in results.items():
        if 'error' in result:
            logger.error(f"\n{name.upper()}: ERROR - {result['error']}")
            continue

        logger.info(f"\n{name.upper()}:")
        logger.info(f"  Confidence: {result.get('confidence', 0):.3f}")
        logger.info(f"  Severity: {result.get('severity', 'unknown')}")

        if name == 'audio' and result.get('has_audio'):
            trans = result.get('transcription', '')
            logger.info(f"  Transcription: {trans[:100]}{'...' if len(trans) > 100 else ''}")

        if name == 'hate_speech' and result.get('text'):
            text = result.get('text', '')
            logger.info(f"  Text: {text}")

        if 'components' in result:
            logger.info(f"  Components: {result['components']}")

def save_results_to_file(results, output_file="analysis_results.json"):
    """Save results to JSON file with collective confidence score"""
    import json
    try:
        # Calculate collective confidence score
        confidence_scores = []
        for detector_name, result in results.items():
            if 'error' not in result and 'confidence' in result:
                confidence_scores.append(result['confidence'])

        collective_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

        # Add collective score to results
        results['collective_confidence'] = collective_confidence

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logging.getLogger(__name__).info(f"Results saved to {output_file}")
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to save results: {e}")

def check_dependencies():
    """Check if required system dependencies are available"""
    import subprocess

    logger = logging.getLogger(__name__)

    # Check ffmpeg
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            logger.warning("ffmpeg not found or not working properly")
        else:
            logger.info("✓ ffmpeg available")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.error("✗ ffmpeg not found. Please install ffmpeg for audio processing.")
        return False

    # Check ffprobe
    try:
        result = subprocess.run(['ffprobe', '-version'], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            logger.warning("ffprobe not found or not working properly")
        else:
            logger.info("✓ ffprobe available")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.error("✗ ffprobe not found. Please install ffmpeg for audio processing.")
        return False

    return True

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Video Content Analyzer')
    parser.add_argument('video_path', nargs='?', help='Path to video file to analyze')
    parser.add_argument('--output', '-o', help='Output folder for frames (default: _frames)')
    parser.add_argument('--results', '-r', help='Output file for results (default: analysis_results.json)')
    parser.add_argument('--frame-rate', '-f', type=int, help='Frame extraction rate in seconds (default: 5)')
    parser.add_argument('--workers', '-w', type=int, help='Max number of worker threads (default: 4)')
    parser.add_argument('--log-file', '-l', help='Log file path (default: video_analyzer.log)')
    parser.add_argument('--frame-width', type=int, help='Frame width (default: 640)')
    parser.add_argument('--frame-height', type=int, help='Frame height (default: 480)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Log level (default: INFO)')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    # Override config with command line args
    if args.video_path:
        Config.VIDEO_PATH = args.video_path
    if args.output:
        Config.OUTPUT_FOLDER = args.output
    if args.frame_rate:
        Config.FRAME_RATE = args.frame_rate
    if args.workers:
        Config.MAX_WORKERS = args.workers
    if args.frame_width:
        Config.FRAME_WIDTH = args.frame_width
    if args.frame_height:
        Config.FRAME_HEIGHT = args.frame_height
    if args.log_level:
        Config.LOG_LEVEL = args.log_level

    # Setup logging
    log_file = args.log_file if args.log_file else "video_analyzer.log"
    logger = setup_logging(log_file)

    # Check dependencies
    if not check_dependencies():
        logger.error("Missing required dependencies. Please install them and try again.")
        sys.exit(1)

    video_path = Config.VIDEO_PATH

    # Validate input
    if not video_path:
        logger.error("No video path provided. Use --help for usage information.")
        sys.exit(1)

    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        sys.exit(1)

    # Validate video file
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video file: {video_path}")
            sys.exit(1)
        cap.release()
    except Exception as e:
        logger.error(f"Error validating video file: {e}")
        sys.exit(1)

    logger.info(f"Starting analysis of: {video_path}")

    try:
        results = analyze_video_with_load_balancing(video_path, Config.OUTPUT_FOLDER)

        results_file = args.results if args.results else "analysis_results.json"
        save_results_to_file(results, results_file)

        logger.info("Analysis completed successfully")

    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Analysis failed with error: {e}")
        sys.exit(1)
