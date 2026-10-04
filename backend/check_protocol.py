import sys, os

PROTOCOL = sys.argv[1] if len(sys.argv) > 1 else "protocols/ASVspoof2019.LA.cm.train.trn.txt"
AUDIO_DIR = sys.argv[2] if len(sys.argv) > 2 else "flac"
EXT = sys.argv[3] if len(sys.argv) > 3 else ".flac"

assert os.path.exists(PROTOCOL), f"[FAIL] protocol not found: {PROTOCOL}"

n_lines = n_bad_cols = n_missing_files = n_bad_label = 0
label_set = set()
col_count_hist = {}

with open(PROTOCOL) as f:
    for i, line in enumerate(f, 1):
        parts = line.strip().split()
        n_lines += 1
        col_count_hist[len(parts)] = col_count_hist.get(len(parts), 0) + 1

        if len(parts) != 5:
            n_bad_cols += 1
            if n_bad_cols <= 3:
                print(f"[WARN] line {i}: expected 5 cols, got {len(parts)} -> {parts}")
            continue

        _, filename, _, _, key = parts
        label_set.add(key)

        if key not in ("bonafide", "spoof"):
            n_bad_label += 1
            if n_bad_label <= 3:
                print(f"[WARN] line {i}: unexpected label '{key}'")

        fpath = os.path.join(AUDIO_DIR, filename + EXT)
        if not os.path.exists(fpath):
            n_missing_files += 1
            if n_missing_files <= 3:
                print(f"[WARN] line {i}: audio file missing -> {fpath}")

print("---")
print(f"total_lines        : {n_lines}")
print(f"column_count_hist  : {col_count_hist}")
print(f"bad_column_rows    : {n_bad_cols}")
print(f"unique_labels_found: {label_set}")
print(f"bad_label_rows     : {n_bad_label}")
print(f"missing_audio_files: {n_missing_files} (checked against dir='{AUDIO_DIR}', ext='{EXT}')")

if n_bad_cols == 0 and n_bad_label == 0 and n_missing_files == 0:
    print("[PASS] protocol clean")
else:
    print("[FAIL] fix above before training")
