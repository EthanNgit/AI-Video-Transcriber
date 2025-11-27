import json
import os
import subprocess
import time
import requests
import soundfile
import torch
from audio_separator.separator import Separator
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OUT_DIR = "ephemeral"
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def post_process_transcripts(segments):
    """Post processes transcript json to catch errors, hallucinations, etc"""

    texts = [seg["text"] for seg in segments]

    prompt = (
        f"""
        You are an expert editor for the Mandarin Chinese dub of SpongeBob SquarePants.
        Your task is to correct an ordered JSON transcript. You will only correct specific character names and key terms to their official spellings.

        RULES:
        1.  **Preserve Original Text:** Your primary rule is to preserve the original text exactly, except for the terms in the 'Official Terms' list.
        2.  **Correct Only from List:** Only change text that is a clear phonetic or visual misspelling of an official term.
        3.  **Maintain JSON Structure:** The output JSON must have the exact same structure as the input.
        4.  **No Other Changes:** Do not fix grammar, add punctuation, or change any other words.
        5.  **Be Precise:** If "谢老板" appears, change it to "蟹老板". If "你好" appears, leave it as "你好".
        7.  When a character name below appears in english in the transcript, replace it with the mandarin version: "Larry" -> "拉里", "Sandy" -> "珊迪"
        6.  **Ending Hallucinations**: You are allowed to remove hallucinations, you must be sure it is one first though,
                Examples of whisper hallucinations:
                    * Youtube style outros, such as: "谢谢大家", "本期视频就先说到这里了,欢迎订阅我的频道哦!", "下次见", "今天就到此为止".
                    * Copyrights/Websites/Links, such as: "example.com", "copyright by ..."
                    * Interpretations of sound, such as: "【海绵宝宝与蟹黄堡王互动声】"
                **Make sure that these hallucinations do not make sense in the context**, if they do not replace with "",
                lines that appear in hallucinations could also just be apart of the transcript

        ---

        # **COMPLETE SPONGEBOB MANDARIN TERM DICTIONARY**

        ## **Main Characters**

        * SpongeBob → 海绵宝宝
        * Patrick → 派大星
        * Squidward → 章鱼哥
        * Plankton → 痞老板
        * Mr. Krabs → 蟹老板
        * Pearl Krabs → 珍珍
        * Sandy Cheeks → 珊迪·奇克斯
        * Mermaid Man → 美人鱼战士
        * Barnacle Boy → 海星少年
        * Gary the Snail → 小蜗
        * Mrs. Puff → 泡芙老师
        * Karen (Plankton's wife) → 凯伦
        * Flying Dutchman → 飞天魔鬼
        * King Neptune → 海神
        * Larry the Lobster → 拉里
        * Bubble Bass → 海霸王
        * Anchovy → 凤尾鱼

        ## **Minor / Recurring Characters**

        * Patrick's Sister → 派大珊
        * Squilliam Fancyson → 章鱼威廉·弗克森
        * Patchy the Pirate → 海盗派奇 *(派斯 is uncommon; 正确为派奇)*
        * Rayman (Man Ray) → 射线恶魔
        * Dirty Bubble → 邪恶泡泡

        ---

        # **Locations (居所)**

        ### Homes

        * the Pineapple House → 菠萝屋 / 凤梨屋
        * Squidward's Moai House → 复活岛人像屋
        * Patrick's Rock House → 石头屋
        * Sandy's Treedome → 圆顶树屋
        * Mr. Krabs' Anchor House → 船锚屋
        * The Chum Bucket → 海之霸 *(痞老板餐厅)*

        ### General Setting

        * Bikini Bottom → 比基尼海滩 / 比奇堡 / 裤头村 *(央配、台配均存在)*
        * The Krusty Krab → 蟹堡王
        * Jellyfish Fields → 水母田
        * Mrs. Puff's Boating School → 泡芙女士的驾驶学校 / 划浪学校 *(版本差异)*

        ### Extra Canon Locations

        * Empty Island (Karate Island) → 空手道圣岛 / 竖笛圣岛
        * New Kelp City → 纽开普市
        * Shell City → 贝壳城
        * Fine Dining Restaurant (Squilliam episodes) → 帆船餐厅

        ---

        # **General Concepts & Creatures**

        * Jellyfish → 水母
        * Clams → 贝壳

        ---

        # **Optional Additions (Useful for correction logic)**

        These appear often in subtitles and may be misheard:

        * 蟹黄堡 (Krabby Patty)
        * 秘密配方 / 神秘配方 (Secret Recipe)
        * 打泡泡 / 泡泡艺术
        * 相机鱼 (generic news reporter fish)
        * 医生鱼 / 警察鱼 / 店员鱼

        ---

        You will be given the full JSON Array of strings. Return only valid JSON. Use double-quoted strings and no trailing commas.

        {json.dumps(texts, ensure_ascii=False)}
       """
    )

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": prompt}]}]
    }

    max_retries = 5
    retry_delay = 2  # seconds

    for attempt in range(max_retries):
        try:
            print(f"Gemini attempt {attempt + 1}/{max_retries}...")
            r = requests.post(url, headers=headers, params={"key": GEMINI_API_KEY}, json=data)
            r.raise_for_status()  # Check for HTTP errors
            out = r.json()

            if "candidates" not in out or not out["candidates"]:
                raise ValueError(f"No candidates returned from Gemini: {out}")

            cand = out["candidates"][0]

            if "content" not in cand:
                raise ValueError(f"No content in candidate: {cand}")

            text_output = cand["content"]["parts"][0].get("text", "").strip()

            # Gemini Issue
            if text_output.startswith("```"):
                # remove the leading ```json or ```
                first_newline = text_output.find("\n")
                text_output = text_output[first_newline + 1:]
                # remove trailing ```
                if text_output.endswith("```"):
                    text_output = text_output[:-3]

            text_output = text_output.strip()
            print(text_output)

            cleaned = json.loads(text_output)

            if len(cleaned) != len(segments):
                raise ValueError(f"LLM returned wrong number of items. Expected {len(segments)}, got {len(cleaned)}")

            result = []
            for seg, new in zip(segments, cleaned):
                s = dict(seg)
                s["text"] = new
                result.append(s)

            return result

        except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Failing.")
                raise e


def get_audio_transcript(input_file: str, prompt: str = None):
    """Gets the transcript for a audio file"""

    client = OpenAI(api_key=OPEN_AI_API_KEY)

    max_retries = 3
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            with open(input_file, "rb") as f:
                resp = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                    language="zh",
                    prompt=prompt
                )
            return resp.model_dump()
        except Exception as e:
            print(f"OpenAI Whisper attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise e


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


def separate_audio_from_video(path_to_video: str):
    """
    Converts a video file to audio file
    :returns directory path where the audio file exists
    """
    file_name = f"{path_to_video.split("/")[-1].split(".")[0]}"
    audio_path = f"{OUT_DIR}/{file_name}/sound/{file_name}.wav"  # Ephemeral file location
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)

    subprocess.run([
        "ffmpeg",
        "-y",  # overwrite output
        "-i", path_to_video,
        "-ac", "1",  # mono
        "-ar", "16000",  # 16 kHz (good for speech + Silero)
        "-vn",  # no video
        audio_path
    ])

    return audio_path


def separate_voice_from_audio(path_to_audio: str):
    """Separates voice from other elements in an audio file"""

    file_name = f"{path_to_audio.split('/')[-1].split('.')[0]}"
    output_directory = f"{OUT_DIR}/{file_name}/vocals"
    os.makedirs(output_directory, exist_ok=True)

    print("Separating vocals... (This may take a while)")
    separator = Separator(
        output_single_stem="Vocals",
        output_dir=output_directory,  # Set the output directory here
        sample_rate=16000
    )

    separator.load_model(model_filename='vocals_mel_band_roformer.ckpt')

    output_files = separator.separate(path_to_audio, {"Vocals": file_name})

    return f"{output_directory}/{output_files[0]}"


def get_audio_vad_metadata(path_to_audio: str):
    """Creates metadata for start and end times of voices"""
    vad_threshold = 0.9  # threshold for voice activity detector
    min_speech_ms = 300  # minimum ms to consider for ?
    min_silence_ms = 500  # minimum silence (in ms) to consider for ?

    # Load Silero VAD
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        trust_repo=True
    )
    (get_speech_timestamps,
     save_audio,
     read_audio,
     VADIterator,
     collect_chunks) = utils

    # Load Audio
    wav, sr = soundfile.read(path_to_audio)

    # Convert stereo → mono if needed
    if wav.ndim == 2:
        wav = wav.mean(axis=1)

    wav = torch.from_numpy(wav).float()

    # Run Silero VAD
    vad = get_speech_timestamps(
        wav,
        model,
        sampling_rate=sr,
        threshold=vad_threshold,
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=min_silence_ms,
        speech_pad_ms=1000
    )
    metadata = [dict(start=(x["start"] / sr), end=(x["end"] / sr)) for x in vad]

    return metadata


def segment_audio_by_vad(path_to_audio: str, vad_metadata):
    """Uses VAD metadata to determine how to merge intervals, slices audio into clips"""
    split_threshold = 3
    intervals = []
    current_start = None
    current_end = None

    file_name = f"{path_to_audio.split('/')[-1].split('.')[0]}"
    clips_path = f"{OUT_DIR}/{file_name}/clips"
    os.makedirs(clips_path, exist_ok=True)

    for seg in vad_metadata:
        start = seg["start"]
        end = seg["end"]

        if current_start is None:
            # start first interval
            current_start = start
            current_end = end
            continue

        gap = start - current_end

        if gap <= split_threshold:
            # merge into current interval
            current_end = max(current_end, end)
        else:
            # close current interval and start a new one
            intervals.append((current_start, current_end))
            current_start = start
            current_end = end

    # append the final interval
    if current_start is not None:
        intervals.append((current_start, current_end))

    print(intervals)

    wav, sr = soundfile.read(path_to_audio)
    slice_paths = []  # store filepaths of created slices

    for idx, (start_sec, end_sec) in enumerate(intervals):
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)

        audio_slice = wav[start_sample:end_sample]

        out_path = f"{clips_path}/{file_name}_slice_{idx:03d}.wav"
        soundfile.write(out_path, audio_slice, sr)
        slice_paths.append((out_path, start_sec, end_sec))

    return slice_paths


def format_timestamp(seconds):
    """Converts seconds to SRT timestamp format (HH:MM:SS,mmm)."""

    total_seconds = int(seconds)
    milliseconds = int((seconds - total_seconds) * 1000)

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


def json_to_srt(json_path, srt_path):
    """Formats transcript json to use correct timestamps"""

    with open(json_path, 'r', encoding='utf-8') as f:
        segments = json.load(f)

    print(segments)

    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(segments, start=1):
            start = segment.get('start', 0)
            end = segment.get('end', 0)
            text = segment.get('text', '').strip()

            if not text:
                continue

            f.write(f"{i}\n")
            f.write(f"{format_timestamp(start)} --> {format_timestamp(end)}\n")
            f.write(f"{text}\n\n")


def overlay_subtitles(video_path, json_path, output_path):
    """Overlays the subtitles from the transcript json on the video"""

    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found: {json_path}")
        return

    # Create a temporary SRT file
    srt_path = "temp_subtitles.srt"
    print(f"Converting {json_path} to SRT format...")
    json_to_srt(json_path, srt_path)

    # FFmpeg command to burn subtitles
    # Using -preset ultrafast for speed as requested
    # Using -y to overwrite output file if it exists
    # Note: 'subtitles' filter requires the path to be escaped properly on Windows if it contains special chars,
    # but for a simple filename in the current dir, it's fine.
    # We use forward slashes for paths in ffmpeg filter to be safe or just relative path.

    # Escape backslashes for Windows ffmpeg filter
    srt_path_escaped = srt_path.replace('\\', '/').replace(':', '\\:')

    cmd = [
        'ffmpeg',
        '-y',
        '-i', video_path,
        '-vf', f"subtitles='{srt_path_escaped}'",
        '-c:a', 'copy',
        '-preset', 'ultrafast',
        output_path
    ]

    print(f"Running FFmpeg to overlay subtitles...")
    print(" ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully created {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error running FFmpeg: {e}")
    except FileNotFoundError:
        print("Error: FFmpeg not found. Please ensure FFmpeg is installed and in your PATH.")
    finally:
        if os.path.exists(srt_path):
            os.remove(srt_path)


def main():
    video_path = "episodes/3a_Jellyfishing.mkv"
    audio_path = separate_audio_from_video(video_path)
    voice_path = separate_voice_from_audio(audio_path)
    md = get_audio_vad_metadata(voice_path)
    clips_path = segment_audio_by_vad(voice_path, md)

    print("Transcribing the audio...")
    all_jsons = []
    for path, start_sec, end_sec in clips_path:
        result = get_audio_transcript(str(path),
                                      "本稿内容与《海绵宝宝》相关，涉及的词汇包括：海绵宝宝、派大星、章鱼哥、痞老板、蟹老板、蟹堡王、蟹黄堡、秘密配方、贝壳。")
        all_jsons.append((result, start_sec, end_sec))

    final_json = merge_whisper_transcripts(all_jsons)

    file_name = f"{video_path.split("/")[-1].split(".")[0]}"
    transcript_path = f"{OUT_DIR}/{file_name}/transcripts/{file_name}_transcripts.json"
    os.makedirs(os.path.dirname(transcript_path), exist_ok=True)

    with open(transcript_path, "w", encoding="utf8") as f:
        json.dump(final_json, f, ensure_ascii=False, indent=2)

    print("Post processing the transcription...")

    corrected_segments = post_process_transcripts(final_json["segments"])

    base, ext = os.path.splitext(transcript_path)
    json_out_path = base + "_post" + ext

    with open(json_out_path, "w", encoding="utf-8") as out:
        json.dump(corrected_segments, out, ensure_ascii=False, indent=2)

    print("Overlaying subtitles..")

    final_video_path = f"{OUT_DIR}/{file_name}/result/{file_name}_with_subtitles.mkv"
    os.makedirs(os.path.dirname(final_video_path), exist_ok=True)
    overlay_subtitles(video_path, json_out_path, final_video_path)


if __name__ == "__main__":
    main()
