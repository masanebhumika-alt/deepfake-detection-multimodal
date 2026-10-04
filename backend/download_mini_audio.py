import os
import soundfile as sf
from datasets import load_dataset

output_dir = r"C:\Deepfake_Detection\backend\data\audio\mini_asv"
flac_dir = os.path.join(output_dir, "flac")
protocol_path = os.path.join(output_dir, "mini_protocol.txt")

os.makedirs(flac_dir, exist_ok=True)

print("Streaming dataset samples from Hugging Face...")

ds = load_dataset("SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA", split="test", streaming=True)

bonafide_count = 0
spoof_count = 0
target_per_class = 1000  # 2,000 total audio samples (~150 MB)

protocol_lines = []

for idx, sample in enumerate(ds):
    # Determine key name for ID dynamically
    file_id = sample.get("utterance_id") or sample.get("id") or sample.get("file_id") or f"LA_E_{idx:07d}"

    label = sample["label"]  # 0 = bonafide, 1 = spoof
    key_str = "bonafide" if label == 0 else "spoof"

    if key_str == "bonafide" and bonafide_count >= target_per_class:
        continue
    if key_str == "spoof" and spoof_count >= target_per_class:
        continue

    audio_data = sample["audio"]["array"]
    sr = sample["audio"]["sampling_rate"]

    out_file = os.path.join(flac_dir, f"{file_id}.flac")
    sf.write(out_file, audio_data, sr)

    protocol_lines.append(f"LA_0000 {file_id} - - {key_str}\n")

    if key_str == "bonafide":
        bonafide_count += 1
    else:
        spoof_count += 1

    if bonafide_count >= target_per_class and spoof_count >= target_per_class:
        break

with open(protocol_path, "w") as f:
    f.writelines(protocol_lines)

print(f"\n[SUCCESS] Saved {bonafide_count} bonafide and {spoof_count} spoof audio files to:")
print(f"Directory: {flac_dir}")
print(f"Protocol: {protocol_path}")