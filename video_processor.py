import json
import os
import subprocess


class VideoProcessor:
    def __init__(self):
        pass

    def overlay_transcription_subtitles(self, path_to_video, path_to_transcription_json, output_path):
        """Overlays the subtitles from the transcript json on the video"""

        if not os.path.exists(path_to_video):
            print(f"Error: Video file not found: {path_to_video}")
            return
        if not os.path.exists(path_to_transcription_json):
            print(f"Error: JSON file not found: {path_to_transcription_json}")
            return

        # Create a temporary SRT file
        srt_path = "temp_subtitles.srt"
        print(f"Converting {path_to_transcription_json} to SRT format...")
        self._json_to_srt(path_to_transcription_json, srt_path)

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
            '-i', path_to_video,
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

    def separate_audio_from_video(self, path_to_video: str, output_path):
        """
        Converts a video file to audio file
        :returns directory path where the audio file exists
        """

        file_name = f"{path_to_video.split("/")[-1].split(".")[0]}"
        audio_path = f"{output_path}/{file_name}.wav"
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

    def _json_to_srt(self, path_to_json, srt_path):
        """Formats transcript json to use correct timestamps"""

        with open(path_to_json, 'r', encoding='utf-8') as f:
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
                f.write(f"{self._format_timestamp(start)} --> {self._format_timestamp(end)}\n")
                f.write(f"{text}\n\n")

    def _format_timestamp(self, seconds):
        """Converts seconds to SRT timestamp format (HH:MM:SS,mmm)."""

        total_seconds = int(seconds)
        milliseconds = int((seconds - total_seconds) * 1000)

        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60

        return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"
