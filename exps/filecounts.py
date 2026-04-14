import os
import librosa
import numpy as np
from pathlib import Path
from collections import defaultdict

def inspect_dataset(root_dir):
    root = Path(root_dir)
    supported = {'.wav', '.mp3', '.m4a', '.mpga'}

    total_files = 0
    total_duration = 0.0
    dataset_summary = {}

    print("=" * 65)
    print(f"  DATASET INSPECTION REPORT")
    print(f"  Root: {root_dir}")
    print("=" * 65)

    for class_folder in sorted(root.iterdir()):
        if not class_folder.is_dir():
            continue

        print(f"\n  CLASS: {class_folder.name}")
        print(f"  {'-' * 55}")

        class_stats = {
            "file_count": 0,
            "formats": defaultdict(int),
            "durations": [],
            "corrupt": [],
            "too_short": [],   # < 1 second
            "silent": [],      # rms energy near zero
        }

        audio_files = [f for f in sorted(class_folder.iterdir())
                       if f.suffix.lower() in supported]

        if not audio_files:
            print(f"    [!] No supported audio files found")
            continue

        for audio_file in audio_files:
            ext = audio_file.suffix.lower()
            class_stats["formats"][ext] += 1

            try:
                y, sr = librosa.load(str(audio_file), sr=None, mono=True)
                duration = librosa.get_duration(y=y, sr=sr)
                rms = np.sqrt(np.mean(y**2))

                class_stats["file_count"] += 1
                class_stats["durations"].append(duration)

                # Flag issues
                if duration < 1.0:
                    class_stats["too_short"].append(audio_file.name)
                if rms < 0.001:
                    class_stats["silent"].append(audio_file.name)

                status = ""
                if duration < 1.0:
                    status = "  [!] TOO SHORT"
                elif rms < 0.001:
                    status = "  [!] SILENT"

                print(f"    {audio_file.name:<45} {duration:6.2f}s{status}")

            except Exception as e:
                class_stats["corrupt"].append(audio_file.name)
                print(f"    {audio_file.name:<45} [CORRUPT: {e}]")

        # Per-class summary
        durations = class_stats["durations"]
        if durations:
            total_dur = sum(durations)
            total_duration += total_dur
            total_files += class_stats["file_count"]

            clips_3s = sum(int(d // 3) for d in durations)

            print(f"\n    --- Summary for {class_folder.name} ---")
            print(f"    Files loaded:       {class_stats['file_count']}")
            print(f"    Formats:            {dict(class_stats['formats'])}")
            print(f"    Total duration:     {total_dur:.1f}s  ({total_dur/60:.1f} min)")
            print(f"    Min duration:       {min(durations):.2f}s")
            print(f"    Max duration:       {max(durations):.2f}s")
            print(f"    Mean duration:      {np.mean(durations):.2f}s")
            print(f"    Estimated 3s clips: {clips_3s}")

            if class_stats["corrupt"]:
                print(f"    [!] Corrupt files:  {class_stats['corrupt']}")
            if class_stats["too_short"]:
                print(f"    [!] Too short (<1s):{class_stats['too_short']}")
            if class_stats["silent"]:
                print(f"    [!] Silent files:   {class_stats['silent']}")

            dataset_summary[class_folder.name] = {
                "files": class_stats["file_count"],
                "total_duration_s": round(total_dur, 2),
                "mean_duration_s": round(np.mean(durations), 2),
                "estimated_3s_clips": clips_3s,
                "corrupt": len(class_stats["corrupt"]),
                "too_short": len(class_stats["too_short"]),
                "silent": len(class_stats["silent"]),
            }

    # Final dataset-level summary
    print("\n" + "=" * 65)
    print("  OVERALL DATASET SUMMARY")
    print("=" * 65)
    print(f"  {'Class':<35} {'Files':>5}  {'Duration':>10}  {'3s clips':>8}  {'Issues':>6}")
    print(f"  {'-'*35}  {'-'*5}  {'-'*10}  {'-'*8}  {'-'*6}")

    total_clips = 0
    for cls, s in dataset_summary.items():
        issues = s["corrupt"] + s["too_short"] + s["silent"]
        dur_str = f"{s['total_duration_s']/60:.1f} min"
        print(f"  {cls:<35} {s['files']:>5}  {dur_str:>10}  {s['estimated_3s_clips']:>8}  {issues:>6}")
        total_clips += s["estimated_3s_clips"]

    print(f"\n  Total files:          {total_files}")
    print(f"  Total duration:       {total_duration/60:.1f} min")
    print(f"  Total 3s clips est.:  {total_clips}")
    print(f"\n  Augmentation needed to reach 100 clips per class:")
    for cls, s in dataset_summary.items():
        clips = s["estimated_3s_clips"]
        if clips >= 100:
            print(f"    {cls:<35} {clips} clips  ->  subsample to 100")
        else:
            factor = round(100 / clips, 1) if clips > 0 else "N/A"
            print(f"    {cls:<35} {clips} clips  ->  need ~{factor}x augmentation")
    print("=" * 65)

    return dataset_summary


if __name__ == "__main__":
    ROOT = r'C:\Users\HP\Downloads\Foss_hack\Data-Pool-Bio-Acoustics-\audio'
    inspect_dataset(ROOT)