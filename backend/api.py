import os
import shutil
import json
import soundfile
from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Optional
import uuid

from corrector import Corrector
from transcriber import Transcriber
from voice_detector import VoiceDetector
from video_processor import VideoProcessor
from dependency_validator import validate_dependencies

# Initialize app
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for serving results
OUT_DIR = "ephemeral"
os.makedirs(OUT_DIR, exist_ok=True)
app.mount("/results", StaticFiles(directory=OUT_DIR), name="results")

# Font configuration
FONT_DIR = "/app/fonts"
FONT_MAP = {
    "en": {
        "NotoSans": os.path.join(FONT_DIR, "NotoSans-Regular.ttf"),
        "Roboto": os.path.join(FONT_DIR, "Roboto-Regular.ttf"),
    },
    "zh": {
        "NotoSansSC": os.path.join(FONT_DIR, "NotoSansSC-Regular.ttf"),
        "NotoSansTC": os.path.join(FONT_DIR, "NotoSansTC-Regular.ttf"),
    }
}

# Initialize components
try:
    validate_dependencies()
    corrector = Corrector()
    transcriber = Transcriber()
    voice_detector = VoiceDetector()
    video_processor = VideoProcessor()
    print("Components initialized")
except Exception as e:
    print(f"Initialization failed: {e}")

@app.get("/fonts")
async def get_fonts():
    """Returns available fonts for each language"""
    return {"fonts": FONT_MAP}

# Helper functions from original main.py
def merge_whisper_transcripts(all_jsons):
    merged_segments = []
    texts = []
    for (json_data, slice_start_sec, slice_end_sec) in all_jsons:
        segs = json_data.get("segments", [])
        for s in segs:
            new_s = s.copy()
            new_s["start"] = s["start"] + slice_start_sec
            new_s["end"] = s["end"] + slice_start_sec
            merged_segments.append(new_s)
            texts.append(new_s["text"])
    merged_segments.sort(key=lambda x: x["start"])
    for i, seg in enumerate(merged_segments):
        seg["id"] = i
    merged = {
        "text": "".join(texts),
        "segments": merged_segments,
        "duration": merged_segments[-1]["end"] if merged_segments else 0,
    }
    return merged

def merge_vad_segments(vad_metadata, split_threshold=3.0):
    if not vad_metadata: return []
    intervals = []
    current_start = vad_metadata[0]["start"]
    current_end = vad_metadata[0]["end"]
    for seg in vad_metadata[1:]:
        gap = seg["start"] - current_end
        if gap <= split_threshold:
            current_end = max(current_end, seg["end"])
        else:
            intervals.append((current_start, current_end))
            current_start = seg["start"]
            current_end = seg["end"]
    intervals.append((current_start, current_end))
    return intervals

def refine_vad_intervals(path_to_audio: str, intervals, padding=0.1):
    refined_intervals = []
    for global_start, global_end in intervals:
        vad_segments = voice_detector.get_audio_vad_metadata(path_to_audio, global_start * 1000, global_end * 1000)
        if not vad_segments:
            continue
        local_start = vad_segments[0]["start"]
        local_end = vad_segments[-1]["end"]
        new_start = max(global_start, global_start + local_start - padding)
        new_end = min(global_end, global_start + local_end + padding)
        refined_intervals.append((new_start, new_end))
    return refined_intervals

def segment_audio_by_vad(path_to_audio: str, output_path, vad_metadata, refine=True):
    file_name = os.path.basename(path_to_audio).split('.')[0]
    intervals = merge_vad_segments(vad_metadata, split_threshold=3.0)
    if refine:
        intervals = refine_vad_intervals(path_to_audio, intervals)
    
    wav, sr = soundfile.read(path_to_audio)
    slice_paths = []
    for idx, (start_sec, end_sec) in enumerate(intervals):
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        audio_slice = wav[start_sample:end_sample]
        out_path = f"{output_path}/{file_name}_slice_{idx:03d}.wav"
        soundfile.write(out_path, audio_slice, sr)
        slice_paths.append((out_path, start_sec, end_sec))
    return slice_paths

@app.post("/transcribe")
async def transcribe_video(
    video: UploadFile = File(...),
    language: str = Form(...),
    font: Optional[str] = Form(None),
    whisper_prompt: Optional[str] = Form(None),
    post_processing: bool = Form(True),
    post_processing_prompt: Optional[str] = Form(None)
):
    session_id = str(uuid.uuid4())
    session_dir = os.path.join(OUT_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    # Save uploaded video
    video_path = os.path.join(session_dir, video.filename)
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    try:
        # 1. Separate Audio
        audio_file_path = os.path.join(session_dir, "sound", "audio.wav")
        os.makedirs(os.path.dirname(audio_file_path), exist_ok=True)
        audio_path = video_processor.separate_audio_from_video(video_path, audio_file_path)

        # 2. Separate Voice
        vocals_dir = os.path.join(session_dir, "vocals", "vocals.wav")
        os.makedirs(os.path.dirname(vocals_dir), exist_ok=True)
        voice_path = voice_detector.separate_voice_from_audio(audio_path, vocals_dir)

        # 3. VAD & Segmentation
        md = voice_detector.get_audio_vad_metadata(voice_path)
        clips_out_path = os.path.join(session_dir, "clips")
        os.makedirs(clips_out_path, exist_ok=True)
        clips_path = segment_audio_by_vad(voice_path, clips_out_path, md)

        # 4. Transcribe
        all_jsons = []
        prompt = whisper_prompt if whisper_prompt else ""
        for path, start_sec, end_sec in clips_path:
            result = transcriber.get_audio_transcript(str(path), prompt)
            all_jsons.append((result, start_sec, end_sec))

        final_json = merge_whisper_transcripts(all_jsons)
        
        transcript_path = os.path.join(session_dir, "transcripts", "transcript.json")
        os.makedirs(os.path.dirname(transcript_path), exist_ok=True)
        with open(transcript_path, "w", encoding="utf8") as f:
            json.dump(final_json, f, ensure_ascii=False, indent=2)

        # 5. Post Processing
        final_transcript_path = transcript_path
        if post_processing:
            # Note: Corrector might need the prompt passed to it if it supports it
            # Assuming corrector.post_process_transcripts uses the prompt if available or default
            # The original code didn't pass the prompt to corrector, so we might need to update Corrector or just use default
            corrected_segments = corrector.post_process_transcripts(final_json["segments"]) # TODO: Pass post_processing_prompt if supported
            
            base, ext = os.path.splitext(transcript_path)
            json_out_path = base + "_post" + ext
            with open(json_out_path, "w", encoding="utf-8") as out:
                json.dump(corrected_segments, out, ensure_ascii=False, indent=2)
            final_transcript_path = json_out_path

        # 6. Overlay Subtitles
        final_video_path = os.path.join(session_dir, "result", "output.mkv")
        os.makedirs(os.path.dirname(final_video_path), exist_ok=True)
        
        # Get font path
        font_path = None
        if font and language in FONT_MAP and font in FONT_MAP[language]:
            font_path = FONT_MAP[language][font]
        
        video_processor.overlay_transcription_subtitles(video_path, final_transcript_path, final_video_path, font_path)

        return {
            "status": "success",
            "video_url": f"/results/{session_id}/result/output.mkv",
            "transcript_url": f"/results/{session_id}/transcripts/{os.path.basename(final_transcript_path)}"
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
