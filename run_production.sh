#!/bin/bash

# Production run script for Video Analyzer
# This script activates the virtual environment and runs the video analyzer

set -e  # Exit on any error

echo "=== Video Analyzer Production Run ==="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if video file is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <video_path> [options]"
    echo "Example: $0 sample_video.mp4 --output results --workers 8"
    exit 1
fi

VIDEO_PATH="$1"
shift  # Remove first argument

# Run the analyzer with provided arguments
echo "Running video analysis on: $VIDEO_PATH"
python main.py "$VIDEO_PATH" "$@"

echo "Analysis completed successfully!"
