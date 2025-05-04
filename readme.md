# Clipper v2.03 - Enhanced AI Highlight Detection

A smart video editing tool that uses AI to automatically detect interesting moments in videos and create clips.

## Features

- **Video Playback**: Load and play MP4, MOV, AVI, MKV video files
- **Audio Support**: Load and analyze MP3, WAV audio files
- **Smart Highlight Detection**: AI-powered detection of emotionally interesting moments
- **Transcription**: Automatic speech-to-text transcription
- **Manual Clip Editing**: Set start and end points for custom clips
- **Clip Preview**: Preview clips before saving
- **Clip Export**: Save clips as MP4 videos
- **Dark Theme**: Eye-friendly dark interface

## Requirements

- Python 3.8 or higher
- See requirements.txt for Python dependencies

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/clipper-v2.git
   cd clipper-v2
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python clipper-v2.03.py
   ```

## Usage

1. Click "Load Video" to open a video or audio file
2. Use the playback controls to navigate through the media
3. To manually create clips:
   - Use "Mark Start" and "Mark End" buttons at points of interest
   - Alternatively, enter timestamps manually
   - Click "Preview Clip" to review
   - Click "Save Clip" to export
4. For AI-powered highlight detection:
   - Click "Transcribe + Detect Highlights"
   - Wait for the analysis to complete
   - Use the navigation buttons to review detected highlights
   - Use "CUT CLIP" to directly save a highlight as a clip

## How It Works

Clipper uses OpenAI's Whisper model for speech recognition and the HuggingFace Transformers library for emotion analysis. The application detects various emotions in speech (joy, surprise, anger, fear, sadness) and automatically identifies potentially interesting moments based on emotional content.

## Limitations

- The tiny Whisper model is used by default for speed, but may have lower accuracy
- Processing large files can take significant time and memory
- Video export requires additional disk space for temporary files

## Credits

- Uses OpenAI's Whisper model for speech recognition
- Uses HuggingFace Transformers for emotion detection
- Built with PyQt5 and MoviePy

## License

This project is licensed under the MIT License - see the LICENSE file for details.