# Whisper Transcription Tool

A Python tool for processing video/audio files to extract and transcribe speech using voice activity detection and OpenAI's Whisper API.

## Work In progress

 - Currently only has hardcoded processing by file, many things are not final.

## Features

- Voice separation from background audio
- Voice Activity Detection (VAD)
- Automated transcription using OpenAI Whisper
- Subtitle overlay on videos
- Post-processing and correction of transcripts

## Requirements

### System Dependencies

- **FFmpeg**: Required for video/audio processing
  - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)
  - Linux: `sudo apt-get install ffmpeg`
  - macOS: `brew install ffmpeg`

### Python Dependencies

See `requirements.txt` for all Python package dependencies.

## Installation

1. Install FFmpeg (see above)
2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:

```
OPEN_AI_API_KEY=your_openai_key_here
GEMINI_API_KEY=your_gemini_key_here
GEMINI_URL=your_gemini_url_here
```

## Attributions and Credits

This project uses several open-source libraries and tools:

### Audio Separator

- **Library**: [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator)
- **License**: MIT License
- **Purpose**: Vocal separation from audio/video files
- **Credits**: This library is built on top of [Ultimate Vocal Remover GUI (UVR)](https://github.com/Anjok07/ultimatevocalremovergui) by Anjok07

### Silero VAD

- **Library**: [Silero VAD](https://github.com/snakers4/silero-vad)
- **License**: MIT License
- **Purpose**: Voice Activity Detection for identifying speech segments in audio
- **Credits**: Developed by Silero Team

### OpenAI Whisper

- **Service**: OpenAI Whisper API
- **Purpose**: Speech-to-text transcription
- **Website**: [OpenAI API](https://openai.com/api/)

### Other Dependencies

- **PyTorch**: Deep learning framework
- **soundfile**: Audio file I/O
- **FFmpeg**: Multimedia framework for video/audio processing

## Usage

Run the dependency validator to check your setup:

```bash
python dependency_validator.py
```

Run the main program:

```bash
python main.py
```

The tool will automatically:

1. Validate all dependencies (FFmpeg, API keys, Python packages)
2. Extract audio from video
3. Separate vocals from background audio
4. Detect voice activity segments
5. Transcribe speech using Whisper
6. Post-process transcripts
7. Generate video with subtitles

## Project Structure

- `main.py` - Main application entry point
- `voice_detector.py` - Voice separation and VAD functionality
- `transcriber.py` - Whisper API transcription
- `corrector.py` - Post-processing and correction
- `video_processor.py` - Video processing and subtitle overlay
- `dependency_validator.py` - Dependency validation utilities
- `README.md` - This file

## License
This project is licensed under the MIT License.

## Contributing

When contributing to this project, please ensure:

1. All dependencies are properly attributed
2. New dependencies are added to `requirements.txt`
3. Attribution comments are added to code using third-party libraries

## Disclaimer

Please ensure you have the necessary rights and permissions for any media files you process with this tool. Respect copyright laws and terms of service for all APIs used.

## Issues and Support

For issues related to:

- **Audio separation**: See [python-audio-separator issues](https://github.com/nomadkaraoke/python-audio-separator/issues)
- **VAD**: See [Silero VAD issues](https://github.com/snakers4/silero-vad/issues)
- **FFmpeg**: See [FFmpeg documentation](https://ffmpeg.org/documentation.html)
- **This project**: Open an issue in this repository
