import nemo.collections.asr as nemo_asr
import torch
import time
import librosa
import soundfile as sf
import os

# Select processing device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load Typhoon ASR Real-Time model
print("Loading Typhoon ASR Real-Time...")
asr_model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="scb10x/typhoon-asr-realtime",
    map_location=device
)

def prepare_audio(input_path, output_path=None, target_sr=16000):
    """
    Prepare audio file for Typhoon ASR Real-Time processing
    """
    if not os.path.exists(input_path):
        print(f"❌ File not found: {input_path}")
        return None

    if output_path is None:
        output_path = "processed_audio.wav"

    try:
        print(f"🎵 Processing: {input_path}")

        # Load and resample audio
        y, sr = librosa.load(input_path, sr=None)
        duration = len(y) / sr

        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            print(f"   Resampled: {sr} Hz → {target_sr} Hz")

        # Normalize audio
        y = y / max(abs(y))

        # Save processed audio
        sf.write(output_path, y, target_sr)
        print(f"Saved: {output_path} ({duration:.1f}s)")
        return output_path

    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# Process your audio file
input_file = "./scam_data/scam4.wav"
processed_file = prepare_audio(input_file)

if processed_file:
    print("🌪️ Running Typhoon ASR Real-Time inference...")

    start_time = time.time()

    # Run transcription
    transcriptions = asr_model.transcribe(audio=[processed_file])

    processing_time = time.time() - start_time

    # Get audio duration for performance calculation
    audio_info = sf.info(processed_file)
    audio_duration = audio_info.duration
    rtf = processing_time / audio_duration

    print(f"⚡ Processing time: {processing_time:.2f}s")
    print(f"🎵 Audio duration: {audio_duration:.2f}s")
    print(f"📊 Real-time factor: {rtf:.2f}x")

    if rtf < 1.0:
        print("🚀 Real-time capable!")
    else:
        print("Batch processing mode")

    try:
        os.remove(processed_file)
        print(f"🗑️ Deleted processed file: {processed_file}")
    except Exception as e:
        print(f"⚠️ Could not delete file {processed_file}: {e}")

else:
    print("❌ No processed audio file available")
    transcriptions = []

if transcriptions:
    print("=" * 50)
    print("📝 TRANSCRIPTION RESULTS")
    print("=" * 50)

    transcription = transcriptions[0]

    print(f"Text: {transcription.text}")

else:
    print("❌ No transcription results available")