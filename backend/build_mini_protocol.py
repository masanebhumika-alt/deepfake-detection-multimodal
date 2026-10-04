import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source_protocol", help="full ASVspoof eval protocol, e.g. ASVspoof2019.LA.cm.eval.trl.txt")
    ap.add_argument("flac_dir", help="dir containing the mini LA_E_*.flac files")
    ap.add_argument("--out", default="data/audio/mini_asv/mini_protocol.txt")
    ap.add_argument("--ext", default=".flac")
    args = ap.parse_args()

    if not os.path.exists(args.source_protocol):
        sys.exit(f"[FAIL] source protocol not found: {args.source_protocol}")
    if not os.path.isdir(args.flac_dir):
        sys.exit(f"[FAIL] flac dir not found: {args.flac_dir}")

    present = {os.path.splitext(f)[0] for f in os.listdir(args.flac_dir) if f.endswith(args.ext)}
    print(f"[INFO] {len(present)} flac files found in {args.flac_dir}")
    if not present:
        sys.exit("[FAIL] flac_dir is empty")

    sample_name = next(iter(present))
    print(f"[INFO] sample filename stem: {sample_name}")

    matched, total, bad_cols, label_counts = 0, 0, 0, {}
    output_dir = os.path.dirname(args.out)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.source_protocol) as fin, open(args.out, "w") as fout:
        for line in fin:
            total += 1
            parts = line.strip().split()
            if len(parts) != 5:
                bad_cols += 1
                continue
            filename = parts[1]
            if filename in present:
                fout.write(line if line.endswith("\n") else line + "\n")
                matched += 1
                label_counts[parts[4]] = label_counts.get(parts[4], 0) + 1

    print(f"[INFO] source protocol total lines : {total}")
    print(f"[INFO] bad-column lines skipped     : {bad_cols}")
    print(f"[INFO] matched rows written         : {matched}")
    print(f"[INFO] label distribution           : {label_counts}")
    print(f"[OUT ] {args.out}")

    if matched == 0:
        print("[FAIL] 0 matches - source_protocol filenames do not correspond to flac_dir contents.")
        print(f"       Check that '{sample_name}' (or similar LA_E_ id) appears as column 2 in source_protocol.")
        print("       You likely pointed at the wrong protocol file (train/dev vs eval).")
        sys.exit(1)


if __name__ == "__main__":
    main()
