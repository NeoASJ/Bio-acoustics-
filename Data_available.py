"""
Western Ghats Frog Audio Dataset Downloader
============================================
Searches iNaturalist API for audio observations of all 70 Western Ghats
frog species and downloads them as a labeled dataset.

Usage:
    pip install requests tqdm
    python western_ghats_frog_audio_downloader.py

Output:
    audio_dataset/
        Clinotarsus_curtipes/
            obs_12345678.mp3
            obs_23456789.mp3
        Duttaphrynus_melanostictus/
            ...
        dataset_summary.csv
"""

import os
import time
import csv
import json
import requests
from pathlib import Path
from typing import Optional, List, Dict
from tqdm import tqdm
from typing import Optional

# ── 70 Western Ghats frog species (scientific name → common name) ────────────
SPECIES = {
    "Clinotarsus curtipes":           "Bicolored Frog",
    "Duttaphrynus melanostictus":     "Common Indian Toad",
    "Duttaphrynus scaber":            "Ferguson's Toad",
    "Euphlyctis aloysii":             "Aloysius Skittering Frog",
    "Euphlyctis cyanophlyctis":       "Common Skittering Frog",
    "Euphlyctis mudigere":            "Mudigere Skittering Frog",
    "Ghatophryne ornata":             "Malabar Torrent Toad",
    "Hoplobatrachus crassus":         "Jerdon's Bull Frog",
    "Hoplobatrachus tigrinus":        "Indian Bull Frog",
    "Hylarana indica":                "Indian Golden Backed Frog",
    "Hylarana intermedius":           "Rao's Intermediate Golden-backed Frog",
    "Hylarana malabarica":            "Fungoid Frog",
    "Indirana diplosticta":           "Malabar Indian Frog",
    "Indirana semipalmata":           "Small-handed Frog",
    "Micrixalus elegans":             "Elegant Dancing Frog",
    "Micrixalus kodayari":            "Kodayar Dancing Frog",
    "Micrixalus kottigeharensis":     "Kottigehar Dancing Frog",
    "Micrixalus niluvasei":           "Niluvase Dancing Frog",
    "Micrixalus uttaraghati":         "Northern Dancing Frog",
    "Microhyla ornata":               "Ornate Narrow Mouthed Frog",
    "Microhyla rubra":                "Red Narrow Mouthed Frog",
    "Minervarya sahyadris":           "Minervarya Frog",
    "Nasikabatrachus sahyadrensis":   "Sahyadri Pig Nosed Frog",
    "Nyctibatrachus beddomii":        "Beddome's Night Frog",
    "Nyctibatrachus dattatreyaensis": "Dattatreya Night Frog",
    "Nyctibatrachus grandis":         "Wayanad Night Frog",
    "Nyctibatrachus jog":             "Jog Night Frog",
    "Nyctibatrachus kempholeyensis":  "Kempholey Night Frog",
    "Nyctibatrachus kumbara":         "Kumbara Night Frog",
    "Nyctibatrachus minimus":         "Miniature Night Frog",
    "Nyctibatrachus petraeus":        "Castle Rock Night Frog",
    "Nyctibatrachus pillaii":         "Pillai's Night Frog",
    "Pedostibes tuberculosus":        "Malabar Tree Toad",
    "Polypedates maculatus":          "Common Tree Frog",
    "Polypedates occidentalis":       "Western Tree Frog",
    "Pseudophilautus amboli":         "Amboli Bush Frog",
    "Pseudophilautus kani":           "Kani Bush Frog",
    "Pseudophilautus wynaadensis":    "Wayanad Bush Frog",
    "Raorchestes agasthyaensis":      "Agasthyamalai Bush Frog",
    "Raorchestes akroparallagi":      "Variable Bush Frog",
    "Raorchestes anili":              "Anil's Bush Frog",
    "Raorchestes beddomii":           "Beddome's Bush Frog",
    "Raorchestes bobingeri":          "Inger's Bush Frog",
    "Raorchestes bombayensis":        "Bombay Bush Frog",
    "Raorchestes chalazodes":         "Granular Bush Frog",
    "Raorchestes charius":            "Seshachar's Bush Frog",
    "Raorchestes chromasynchysi":     "Confusing Colored Bush Frog",
    "Raorchestes glandulosus":        "Glandular Bush Frog",
    "Raorchestes griet":              "Griet's Bush Frog",
    "Raorchestes jayarami":           "Jayaram's Bush Frog",
    "Raorchestes johnceei":           "Johncee's Bush Frog",
    "Raorchestes kakachi":            "Kakachi Bush Frog",
    "Raorchestes luteolus":           "Yellow Bush Frog",
    "Raorchestes manohari":           "Manohar's Bush Frog",
    "Raorchestes nerostagona":        "Kalpetta Yellow Bush Frog",
    "Raorchestes ochlandrae":         "Ochlandra Reed Bush Frog",
    "Raorchestes ponmudi":            "Ponmudi Bush Frog",
    "Raorchestes tuberohumerus":      "Knob Handed Bush Frog",
    "Rhacophorus calcadensis":        "Kalakkad Tree Frog",
    "Rhacophorus lateralis":          "Small Tree Frog",
    "Rhacophorus malabaricus":        "Malabar Gliding Frog",
    "Sphaerotheca breviceps":         "Indian Burrowing Frog",
    "Sphaerotheca dobsonii":          "Dobson's Burrowing Frog",
    "Uperedon triangularis":          "Triangular Balloon Frog",
    "Uperedon mormoratus":            "Rao's Marbled Balloon Frog",
    "Uperedon taprobanicus":          "Sri Lankan Balloon Frog",
    "Uperedon variegatus":            "Variegated Balloon Frog",
    "Zakarana caparata":              "Wrinkled Cricket Frog",
    "Zakerana mudduraja":             "Mudduraja's Cricket Frog",
    "Zakerana rufescens":             "Red Cricket Frog",
}

INAT_API   = "https://api.inaturalist.org/v1"
OUTPUT_DIR = Path("audio_dataset")
MAX_PER_SPECIES = 50          # cap per species (raise if you want more)
DELAY_SECONDS   = 1.0         # polite delay between API calls


# ── helpers ──────────────────────────────────────────────────────────────────

def get_taxon_id(scientific_name: str) -> Optional[int]:
    """Resolve scientific name → iNaturalist taxon ID."""
    r = requests.get(
        f"{INAT_API}/taxa",
        params={"q": scientific_name, "rank": "species", "per_page": 1},
        timeout=15,
    )
    r.raise_for_status()
    results = r.json().get("results", [])
    if results:
        return results[0]["id"]
    return None


def fetch_audio_observations(taxon_id: int, per_page: int = 50) -> List[Dict]:
    """Return observations that have at least one sound file."""
    observations = []
    page = 1
    while len(observations) < MAX_PER_SPECIES:
        r = requests.get(
            f"{INAT_API}/observations",
            params={
                "taxon_id":   taxon_id,
                "sounds":     "true",       # only observations WITH audio
                "quality_grade": "research",
                "per_page":   min(per_page, MAX_PER_SPECIES - len(observations)),
                "page":       page,
                "order_by":   "votes",
            },
            timeout=15,
        )
        r.raise_for_status()
        data   = r.json()
        batch  = data.get("results", [])
        if not batch:
            break
        observations.extend(batch)
        if len(batch) < per_page:
            break
        page += 1
        time.sleep(DELAY_SECONDS)
    return observations


def extract_sound_urls(obs: dict) -> List[str]:
    """Pull direct mp3/wav URLs from an observation dict."""
    urls = []
    for sound in obs.get("sounds", []):
        url = sound.get("file_url") or sound.get("file")
        if url:
            urls.append(url)
    return urls


def download_file(url: str, dest: Path) -> bool:
    """Download a single file; return True on success."""
    try:
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return True
    except Exception as e:
        print(f"    ⚠  Download failed ({url}): {e}")
        return False


# ── main pipeline ─────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    summary_rows = []

    print(f"\n{'='*60}")
    print("  Western Ghats Frog Audio Dataset Downloader")
    print(f"  {len(SPECIES)} species  ·  max {MAX_PER_SPECIES} clips each")
    print(f"  Output → {OUTPUT_DIR.resolve()}")
    print(f"{'='*60}\n")

    for sci_name, common_name in SPECIES.items():
        safe_dir = OUTPUT_DIR / sci_name.replace(" ", "_")
        safe_dir.mkdir(exist_ok=True)

        print(f"🐸  {sci_name}  ({common_name})")

        # 1. Resolve taxon ID
        taxon_id = get_taxon_id(sci_name)
        if not taxon_id:
            print(f"   ✗ Taxon not found on iNaturalist — skipping\n")
            summary_rows.append({
                "scientific_name": sci_name,
                "common_name":     common_name,
                "taxon_id":        "",
                "observations":    0,
                "audio_files":     0,
                "status":          "taxon_not_found",
            })
            time.sleep(DELAY_SECONDS)
            continue

        # 2. Fetch observations with audio
        time.sleep(DELAY_SECONDS)
        try:
            observations = fetch_audio_observations(taxon_id)
        except Exception as e:
            print(f"   ✗ API error: {e}\n")
            summary_rows.append({
                "scientific_name": sci_name,
                "common_name":     common_name,
                "taxon_id":        taxon_id,
                "observations":    0,
                "audio_files":     0,
                "status":          f"api_error: {e}",
            })
            continue

        print(f"   Found {len(observations)} research-grade observations with audio")

        # 3. Download each sound file
        downloaded = 0
        for obs in observations:
            obs_id    = obs["id"]
            sound_urls = extract_sound_urls(obs)
            for idx, url in enumerate(sound_urls):
                ext  = Path(url.split("?")[0]).suffix or ".mp3"
                dest = safe_dir / f"obs_{obs_id}_{idx}{ext}"
                if dest.exists():
                    downloaded += 1
                    continue
                if download_file(url, dest):
                    downloaded += 1
                time.sleep(0.3)   # brief pause per file

        print(f"   ✓ Downloaded {downloaded} audio file(s) → {safe_dir}\n")
        summary_rows.append({
            "scientific_name": sci_name,
            "common_name":     common_name,
            "taxon_id":        taxon_id,
            "observations":    len(observations),
            "audio_files":     downloaded,
            "status":          "ok" if downloaded > 0 else "no_audio_found",
        })

    # 4. Write summary CSV
    summary_path = OUTPUT_DIR / "dataset_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "scientific_name", "common_name", "taxon_id",
            "observations", "audio_files", "status"
        ])
        writer.writeheader()
        writer.writerows(summary_rows)

    total_files = sum(r["audio_files"] for r in summary_rows)
    found       = sum(1 for r in summary_rows if r["audio_files"] > 0)
    print(f"\n{'='*60}")
    print(f"  ✅  Done!  {total_files} audio files across {found}/{len(SPECIES)} species")
    print(f"  📄  Summary saved → {summary_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()