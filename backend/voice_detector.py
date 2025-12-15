import os
from typing import Any

import soundfile
import torch
from audio_separator.separator import Separator


class VoiceDetector:
    def __init__(self):
        pass

    def separate_voice_from_audio(self, path_to_audio: str, output_path: str):
        """Separates voice from other elements in an audio file. Skips separation for short clips (<30s)."""

        out_dir = os.path.dirname(output_path)
        out_name = os.path.splitext(os.path.basename(output_path))[0]

        os.makedirs(out_dir, exist_ok=True)

        # Read audio to check length
        wav, sr = soundfile.read(path_to_audio)
        audio_duration_sec = len(wav) / sr
        
        # Skip separation for short audio clips
        if audio_duration_sec < 30:
            print(f"Audio is {audio_duration_sec:.1f}s (< 30s). Skipping voice separation, using original audio.")
            result_path = os.path.join(out_dir, out_name + ".wav")
            
            # Convert to mono if needed
            if wav.ndim == 2:
                wav = wav.mean(axis=1)
            soundfile.write(result_path, wav, sr)
            return result_path

        print("Separating vocals... (This may take a while)")

        separator = Separator(
            output_single_stem="Vocals",
            sample_rate=16000,
            output_dir=out_dir
        )

        separator.load_model(model_filename='vocals_mel_band_roformer.ckpt')

        output_files = separator.separate(path_to_audio, {"Vocals": out_name})
        result_path = os.path.join(out_dir, output_files[0])

        # Convert to mono if needed
        wav, sr = soundfile.read(result_path)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
            soundfile.write(result_path, wav, sr)

        return result_path

    def get_audio_vad_metadata(self, path_to_audio: str, start_ms: int = 0, end_ms: int = float('inf')) -> list[dict[str, Any]]:
        """Creates metadata for start and end times of voices."""

        wav, sr = soundfile.read(path_to_audio)

        total_duration_ms = (len(wav) / sr) * 1000
        if end_ms == float('inf') or end_ms is None:
            end_ms = total_duration_ms

        if start_ms < 0:
            raise ValueError(f"Start time ({start_ms}ms) cannot be negative.")
        
        if end_ms > total_duration_ms:
            raise ValueError(f"End time ({end_ms}ms) exceeds audio duration ({total_duration_ms:.2f}ms).")

        if start_ms >= end_ms:
            raise ValueError(f"Start time ({start_ms}ms) must be less than end time ({end_ms}ms).")

        start_sample = int((start_ms / 1000) * sr)
        end_sample = int((end_ms / 1000) * sr)
        
        segment_audio = wav[start_sample:end_sample]

        return self._run_vad(segment_audio)
    
    def _run_vad(self, audio, threshold=0.9, min_speech_ms=300, min_silence_ms=500, speech_pad_ms=1000) -> list[dict[str, Any]]:
        """
        Core VAD function that runs on audio tensor/array.

        Args:
            audio: numpy array or torch tensor of audio samples
            sr: sample rate
            threshold: VAD confidence threshold
            min_speech_ms: minimum speech duration in ms
            min_silence_ms: minimum silence duration in ms
            speech_pad_ms: padding around speech in ms

        Returns:
            List of dicts with 'start' and 'end' in seconds
        """
        vad_model, vad_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )

        get_speech_timestamps = vad_utils[0]

        if isinstance(audio, torch.Tensor):
            wav = audio
        else:
            wav = torch.from_numpy(audio).float()

        vad_result = get_speech_timestamps(
            wav,
            vad_model,
            sampling_rate=16000,
            threshold=threshold,
            min_speech_duration_ms=min_speech_ms,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms
        )

        return [{"start": x["start"] / 16000, "end": x["end"] / 16000} for x in vad_result]