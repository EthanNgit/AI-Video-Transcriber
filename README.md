# Video Transcription Tool

Web-Based tool for generating the transcript of videos. Uses OpenAi Whisper 2 model along with a couple optimization steps in order to more accurately transcribe audio. Optionally can use Gemini LLM to post process the transcripts to reduce the hallucinations on Whisper's behalf as well as correct content specific mistakes such as names.

## Features

- **Web Interface**: Modern, intuitive UI for video upload and transcription
- **Multi-language Support**: English and Chinese transcription with language-specific fonts
- **Voice Activity Detection**: Intelligent segmentation of speech from silence
- **Voice Separation**: Automatic isolation of vocals from background audio
- **Customizable Transcription**: Optional Whisper prompts for context-aware transcription
- **Post-Processing**: LLM-based transcript refinement and correction
- **Automatic Subtitles**: Hardcoded subtitles on output video with font selection
- **Dockerized**: Fully containerized application with all dependencies included

## Requirements

### System Dependencies

- **Docker**: Required for running the application
  - Windows/macOS: [Docker Desktop](https://www.docker.com/products/docker-desktop/)
  - Linux: [Docker Engine](https://docs.docker.com/engine/install/)
- **Docker Compose**: Included with Docker Desktop, or install separately on Linux

**Note**: FFmpeg and all other dependencies are handled inside the Docker containers - no local installation required!

## Quick Start

1. **Clone the repository**

2. **Set up environment variables**

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_key_here
GEMINI_API_KEY=your_gemini_key_here
GEMINI_URL=your_gemini_url_here
```

3. **Start the application**

```bash
docker compose up --build
```

4. **Access the web interface**

Open your browser and navigate to: `http://localhost:8080`

## Key Libraries & Services

- **[python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator)** - Vocal separation (built on [Ultimate Vocal Remover GUI](https://github.com/Anjok07/ultimatevocalremovergui) by Anjok07)
- **[Silero VAD](https://github.com/snakers4/silero-vad)** - Voice Activity Detection
- **OpenAI Whisper API** - Speech-to-text transcription
- **FFmpeg** - Video/audio processing and subtitle rendering
- **ONNX Runtime** - AI model inference
- **Noto Sans Fonts** - Multi-language subtitle support (Google Fonts, OFL license)

## Usage

### Configuration

Environment Variables:

- `OPENAI_API_KEY` - OpenAI API key for Whisper transcription (required)
- `GEMINI_API_KEY` - Google Gemini API key for post-processing (optional)
- `GEMINI_URL` - Gemini API endpoint URL (optional)

### Web Interface

1. Upload a video file (any common format: MP4, MKV, AVI, etc.)
2. Select the language (English or Chinese)
3. Choose a subtitle font appropriate for the language
4. Optionally provide a Whisper prompt for better transcription context
5. Toggle post-processing on/off and provide instructions if needed
6. Click "Transcribe" and wait for processing to complete
7. Download the transcribed video with hardcoded subtitles and/or the transcript JSON

### Processing Pipeline

The tool automatically:

1. Validates all dependencies (handled in Docker)
2. Extracts audio from the uploaded video
3. Separates vocals from the audio
4. Detects voice activity segments to skip silence
5. Transcribes speech using OpenAI Whisper API
6. Optionally post-processes transcripts with LLM refinement
7. Generates video with hardcoded subtitles using selected font and transcripts json

### Performance Notes

- The backend uses CPU-only PyTorch to ensure universal compatibility
- First build may take several minutes due to dependency downloads
- Subsequent builds use Docker layer caching for faster rebuilds
- Processing time depends on video length and hardware

## License

This project is licensed under the MIT License.
