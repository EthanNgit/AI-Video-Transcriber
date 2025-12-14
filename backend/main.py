"""
Whisper Transcription Tool

A tool for processing video/audio files to extract and transcribe speech.

Attribution & Credits:
- Audio separation: python-audio-separator (https://github.com/nomadkaraoke/python-audio-separator)
  Built on Ultimate Vocal Remover GUI by Anjok07 (https://github.com/Anjok07/ultimatevocalremovergui)
- Voice Activity Detection: Silero VAD (https://github.com/snakers4/silero-vad)
- Transcription: OpenAI Whisper API (https://openai.com/api/)
- Video processing: FFmpeg (https://ffmpeg.org/)
"""

import json
import os
import sys
import soundfile
from dotenv import load_dotenv

from corrector import Corrector
from transcriber import Transcriber
from voice_detector import VoiceDetector
from video_processor import VideoProcessor
from dependency_validator import validate_dependencies

load_dotenv()
OUT_DIR = "ephemeral"


def merge_whisper_transcripts(all_jsons):
    """Merges all related whisper requests into one transcript json"""

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
    """Merges VAD segments that are within split_threshold seconds of each other."""
    if not vad_metadata:
        return []   

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
    """Second VAD pass to refine intervals by trimming leading/trailing silence."""
    refined_intervals = []

    for global_start, global_end in intervals:
        vad_segments = voice_detector.get_audio_vad_metadata(path_to_audio, global_start * 1000, global_end * 1000)

        if not vad_segments:
            print(f"No speech in interval {global_start:.2f}-{global_end:.2f}, skipping")
            continue

        local_start = vad_segments[0]["start"]
        local_end = vad_segments[-1]["end"]

        new_start = max(global_start, global_start + local_start - padding)
        new_end = min(global_end, global_start + local_end + padding)

        trimmed_start = max(0, local_start - padding)
        trimmed_end = (global_end - global_start) - local_end - padding

        if trimmed_start > 0.3 or trimmed_end > 0.3:
            print(f"Refined {global_start:.2f}-{global_end:.2f} -> {new_start:.2f}-{new_end:.2f} "
                  f"(trimmed {trimmed_start:.2f}s start, {max(0, trimmed_end):.2f}s end)")

        refined_intervals.append((new_start, new_end))

    return refined_intervals


def segment_audio_by_vad(path_to_audio: str, output_path, vad_metadata, refine=True):
    """Uses VAD metadata to merge intervals, optionally refine, then slice audio into clips."""
    file_name = path_to_audio.split('/')[-1].split('.')[0]

    # First pass: merge nearby segments
    intervals = merge_vad_segments(vad_metadata, split_threshold=3.0)
    print(f"First pass: {len(intervals)} intervals")

    # Second pass: refine each interval
    if refine:
        print("Running second VAD pass to refine intervals...")
        intervals = refine_vad_intervals(path_to_audio, intervals)
        print(f"After refinement: {len(intervals)} intervals")

    print(intervals)

    # Create clips
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


def main(video_processor, voice_detector, transcriber, corrector):
    video_path = "episodes/ported/17b_RockBottom.mkv"
    file_name = f"{video_path.split("/")[-1].split(".")[0]}"

    audio_file_path = f"{OUT_DIR}/{file_name}/sound/{file_name}_audio.wav"
    os.makedirs(os.path.dirname(audio_file_path), exist_ok=True)
    audio_path = video_processor.separate_audio_from_video(video_path, audio_file_path)

    vocals_dir = f"{OUT_DIR}/{file_name}/vocals/{file_name}_vocals.wav"
    os.makedirs(os.path.dirname(vocals_dir), exist_ok=True)
    voice_path = voice_detector.separate_voice_from_audio(audio_path, vocals_dir)

    md = voice_detector.get_audio_vad_metadata(voice_path)

    clips_out_path = f"{OUT_DIR}/{file_name}/clips"
    os.makedirs(clips_out_path, exist_ok=True)
    clips_path = segment_audio_by_vad(voice_path, clips_out_path, md)

    print("Transcribing the audio...")
    all_jsons = []
    for path, start_sec, end_sec in clips_path:
        result = transcriber.get_audio_transcript(str(path),
                                                  "本稿内容与《海绵宝宝》相关，涉及的词汇包括：海绵宝宝、派大星、章鱼哥、痞老板、蟹老板、蟹堡王、蟹黄堡、秘密配方、贝壳。")
        # Add debug logging
        print(f"Clip {path}: offset={start_sec:.2f}s")
        for seg in result.get("segments", []):
            print(
                f"  Local: {seg['start']:.2f}-{seg['end']:.2f} | Global: {seg['start'] + start_sec:.2f}-{seg['end'] + start_sec:.2f} | {seg['text'][:30]}")
        all_jsons.append((result, start_sec, end_sec))

    final_json = merge_whisper_transcripts(all_jsons)

    transcript_path = f"{OUT_DIR}/{file_name}/transcripts/{file_name}_transcripts.json"
    os.makedirs(os.path.dirname(transcript_path), exist_ok=True)

    with open(transcript_path, "w", encoding="utf8") as f:
        json.dump(final_json, f, ensure_ascii=False, indent=2)

    print("Post processing the transcription...")

    corrected_segments = corrector.post_process_transcripts(final_json["segments"])

    base, ext = os.path.splitext(transcript_path)
    json_out_path = base + "_post" + ext

    with open(json_out_path, "w", encoding="utf-8") as out:
        json.dump(corrected_segments, out, ensure_ascii=False, indent=2)

    print("Overlaying subtitles..")
    final_video_path = f"{OUT_DIR}/{file_name}/result/{file_name}_with_subtitles.mkv"
    os.makedirs(os.path.dirname(final_video_path), exist_ok=True)
    video_processor.overlay_transcription_subtitles(video_path, json_out_path, final_video_path)


if __name__ == "__main__":
    # Validate all dependencies before running
    print("=" * 60)
    print("Validating dependencies...")
    print("=" * 60)
    validate_dependencies()
    print("=" * 60)
    print()
    
    # Initialize components after validation
    print("Initializing components...")
    try:
        corrector = Corrector()
        transcriber = Transcriber()
        voice_detector = VoiceDetector()
        video_processor = VideoProcessor()
        print("All components initialized successfully\n")
    except Exception as e:
        print(f"Failed to initialize components: {e}")
        sys.exit(1)
    
    # Run main program
    main(video_processor, voice_detector, transcriber, corrector)
