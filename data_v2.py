"""
Western Ghats Frog Audio Dataset Downloader v2
===============================================
Fixes:
  - Corrected taxonomy for 8 species that failed ("Taxon not found")
  - Adds Xeno-canto as a second audio source for species with 0 iNaturalist audio
  - Python 3.9 compatible

Sources:
  1. iNaturalist  — research-grade observations with sound
  2. Xeno-canto   — frog call recordings (xenocanto.org)

Usage:
    pip install requests tqdm
    python western_ghats_frog_audio_downloader_v2.py

Output folder structure:
    audio_dataset/
        Clinotarsus_curtipes/
            inat_obs_12345678_0.mp3
            xc_123456.mp3
        ...
        dataset_summary.csv
"""

import os
import time
import csv
import requests
from pathlib import Path
from typing import Optional, List, Dict

# ── Taxonomy corrections ──────────────────────────────────────────────────────
# iNaturalist uses the MOST CURRENT accepted name.
# Several species in your list use older synonyms — corrected below.

SPECIES = {
    # scientific name on iNaturalist         : common name
    "Clinotarsus curtipes":                    "Bicolored Frog",
    "Duttaphrynus melanostictus":              "Common Indian Toad",
    "Duttaphrynus scaber":                     "Ferguson's Toad",
    "Euphlyctis aloysii":                      "Aloysius Skittering Frog",
    "Euphlyctis cyanophlyctis":                "Common Skittering Frog",
    "Euphlyctis mudigere":                     "Mudigere Skittering Frog",
    "Ghatophryne ornata":                      "Malabar Torrent Toad",
    "Hoplobatrachus crassus":                  "Jerdon's Bull Frog",
    "Hoplobatrachus tigerinus":                "Indian Bull Frog",           # FIXED: tigrinus → tigerinus
    "Indosylvirana indica":                    "Indian Golden Backed Frog",  # FIXED: Hylarana indica → Indosylvirana indica
    "Indosylvirana intermedia":                "Rao's Intermediate Golden-backed Frog",  # FIXED: Hylarana intermedius
    "Hylarana malabarica":                     "Fungoid Frog",
    "Indirana diplosticta":                    "Malabar Indian Frog",
    "Indirana semipalmata":                    "Small-handed Frog",
    "Micrixalus elegans":                      "Elegant Dancing Frog",
    "Micrixalus kodayari":                     "Kodayar Dancing Frog",
    "Micrixalus kottigeharensis":              "Kottigehar Dancing Frog",
    "Micrixalus niluvasei":                    "Niluvase Dancing Frog",
    "Micrixalus uttaraghati":                  "Northern Dancing Frog",
    "Microhyla ornata":                        "Ornate Narrow Mouthed Frog",
    "Microhyla rubra":                         "Red Narrow Mouthed Frog",
    "Minervarya sahyadris":                    "Minervarya Frog",
    "Nasikabatrachus sahyadrensis":            "Sahyadri Pig Nosed Frog",
    "Nyctibatrachus beddomii":                 "Beddome's Night Frog",
    "Nyctibatrachus dattatreyaensis":          "Dattatreya Night Frog",
    "Nyctibatrachus grandis":                  "Wayanad Night Frog",
    "Nyctibatrachus jog":                      "Jog Night Frog",
    "Nyctibatrachus kempholeyensis":           "Kempholey Night Frog",
    "Nyctibatrachus kumbara":                  "Kumbara Night Frog",
    "Nyctibatrachus minimus":                  "Miniature Night Frog",
    "Nyctibatrachus petraeus":                 "Castle Rock Night Frog",
    "Nyctibatrachus pillaii":                  "Pillai's Night Frog",
    "Pedostibes tuberculosus":                 "Malabar Tree Toad",
    "Polypedates maculatus":                   "Common Tree Frog",
    "Polypedates occidentalis":                "Western Tree Frog",
    "Pseudophilautus amboli":                  "Amboli Bush Frog",
    "Pseudophilautus kani":                    "Kani Bush Frog",
    "Pseudophilautus wynaadensis":             "Wayanad Bush Frog",
    "Raorchestes agasthyaensis":               "Agasthyamalai Bush Frog",
    "Raorchestes akroparallagi":               "Variable Bush Frog",
    "Raorchestes anili":                       "Anil's Bush Frog",
    "Raorchestes beddomii":                    "Beddome's Bush Frog",
    "Raorchestes bobingeri":                   "Inger's Bush Frog",
    "Raorchestes bombayensis":                 "Bombay Bush Frog",
    "Raorchestes chalazodes":                  "Granular Bush Frog",
    "Raorchestes charius":                     "Seshachar's Bush Frog",
    "Raorchestes chromasynchysi":              "Confusing Colored Bush Frog",
    "Raorchestes glandulosus":                 "Glandular Bush Frog",
    "Raorchestes griet":                       "Griet's Bush Frog",
    "Raorchestes jayarami":                    "Jayaram's Bush Frog",
    "Raorchestes johnceei":                    "Johncee's Bush Frog",
    "Raorchestes kakachi":                     "Kakachi Bush Frog",
    "Raorchestes luteolus":                    "Yellow Bush Frog",
    "Raorchestes manohari":                    "Manohar's Bush Frog",
    "Raorchestes nerostagona":                 "Kalpetta Yellow Bush Frog",
    "Raorchestes ochlandrae":                  "Ochlandra Reed Bush Frog",
    "Raorchestes ponmudi":                     "Ponmudi Bush Frog",
    "Raorchestes tuberohumerus":               "Knob Handed Bush Frog",
    "Rhacophorus calcadensis":                 "Kalakkad Tree Frog",
    "Rhacophorus lateralis":                   "Small Tree Frog",
    "Rhacophorus malabaricus":                 "Malabar Gliding Frog",
    "Sphaerotheca breviceps":                  "Indian Burrowing Frog",
    "Sphaerotheca dobsonii":                   "Dobson's Burrowing Frog",
    "Uperodon triangularis":                   "Triangular Balloon Frog",    # FIXED: Uperedon → Uperodon
    "Uperodon mormoratus":                     "Rao's Marbled Balloon Frog", # FIXED: Uperedon → Uperodon
    "Uperodon taprobanicus":                   "Sri Lankan Balloon Frog",    # FIXED: Uperedon → Uperodon
    "Uperodon variegatus":                     "Variegated Balloon Frog",    # FIXED: Uperedon → Uperodon
    "Euphlyctis caparata":                     "Wrinkled Cricket Frog",      # FIXED: Zakarana → Euphlyctis
    "Zakerana mudduraja":                      "Mudduraja's Cricket Frog",
    "Zakerana rufescens":                      "Red Cricket Frog",
}

INAT_API        = "https://api.inaturalist.org/v1"
XC_API          = "https://www.xeno-canto.org/api/2/recordings"
OUTPUT_DIR      = Path("audio_dataset")
MAX_INAT        = 50    # max clips per species from iNaturalist
MAX_XC          = 20    # max clips per species from Xeno-canto
DELAY           = 1.0   # seconds between API calls


# ═══════════════════════════════════════════════════════════
#  iNaturalist helpers
# ═══════════════════════════════════════════════════════════

def inat_get_taxon_id(name: str) -> Optional[int]:
    r = requests.get(f"{INAT_API}/taxa",
                     params={"q": name, "rank": "species", "per_page": 1},
                     timeout=15)
    r.raise_for_status()
    results = r.json().get("results", [])
    return results[0]["id"] if results else None


def inat_fetch_audio_obs(taxon_id: int) -> List[Dict]:
    obs, page = [], 1
    while len(obs) < MAX_INAT:
        r = requests.get(f"{INAT_API}/observations",
                         params={
                             "taxon_id":     taxon_id,
                             "sounds":       "true",
                             "quality_grade":"research",
                             "per_page":     min(50, MAX_INAT - len(obs)),
                             "page":         page,
                             "order_by":     "votes",
                         }, timeout=15)
        r.raise_for_status()
        batch = r.json().get("results", [])
        if not batch:
            break
        obs.extend(batch)
        if len(batch) < 50:
            break
        page += 1
        time.sleep(DELAY)
    return obs


def inat_sound_urls(obs: dict) -> List[str]:
    return [s.get("file_url") or s.get("file", "")
            for s in obs.get("sounds", []) if s.get("file_url") or s.get("file")]


# ═══════════════════════════════════════════════════════════
#  Xeno-canto helpers
# ═══════════════════════════════════════════════════════════

def xc_search(name: str) -> List[Dict]:
    """Search Xeno-canto for frog calls by scientific name."""
    try:
        r = requests.get(XC_API,
                         params={"query": name},
                         timeout=15)
        r.raise_for_status()
        data = r.json()
        return data.get("recordings", [])[:MAX_XC]
    except Exception as e:
        print(f"   ⚠  Xeno-canto search failed: {e}")
        return []


def xc_download(recordings: List[Dict], dest_dir: Path) -> int:
    """Download Xeno-canto recordings; return count downloaded."""
    downloaded = 0
    for rec in recordings:
        file_url = rec.get("file", "")
        if not file_url:
            continue
        if not file_url.startswith("http"):
            file_url = "https:" + file_url
        xc_id = rec.get("id", "unknown")
        dest  = dest_dir / f"xc_{xc_id}.mp3"
        if dest.exists():
            downloaded += 1
            continue
        try:
            resp = requests.get(file_url, stream=True, timeout=30)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            downloaded += 1
        except Exception as e:
            print(f"    ⚠  XC download failed ({file_url}): {e}")
        time.sleep(0.4)
    return downloaded


# ═══════════════════════════════════════════════════════════
#  Generic file downloader
# ═══════════════════════════════════════════════════════════

def download_file(url: str, dest: Path) -> bool:
    try:
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return True
    except Exception as e:
        print(f"    ⚠  Download failed ({url}): {e}")
        return False


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    summary_rows = []

    print(f"\n{'='*65}")
    print("  Western Ghats Frog Audio Dataset Downloader  v2")
    print(f"  {len(SPECIES)} species  |  iNaturalist + Xeno-canto")
    print(f"  Output → {OUTPUT_DIR.resolve()}")
    print(f"{'='*65}\n")

    for sci_name, common_name in SPECIES.items():
        safe_dir = OUTPUT_DIR / sci_name.replace(" ", "_")
        safe_dir.mkdir(exist_ok=True)
        print(f"🐸  {sci_name}  ({common_name})")

        inat_count = 0
        xc_count   = 0

        # ── iNaturalist ───────────────────────────────────────────
        taxon_id = inat_get_taxon_id(sci_name)
        if taxon_id:
            time.sleep(DELAY)
            try:
                observations = inat_fetch_audio_obs(taxon_id)
                print(f"   iNat: {len(observations)} research-grade audio obs found")
                for obs in observations:
                    obs_id = obs["id"]
                    for idx, url in enumerate(inat_sound_urls(obs)):
                        ext  = Path(url.split("?")[0]).suffix or ".mp3"
                        dest = safe_dir / f"inat_obs_{obs_id}_{idx}{ext}"
                        if dest.exists():
                            inat_count += 1
                            continue
                        if download_file(url, dest):
                            inat_count += 1
                        time.sleep(0.3)
            except Exception as e:
                print(f"   iNat API error: {e}")
        else:
            print(f"   iNat: taxon not found (check name)")

        # ── Xeno-canto (always try, regardless of iNat result) ────
        time.sleep(DELAY)
        xc_recs = xc_search(sci_name)
        if xc_recs:
            print(f"   XC:   {len(xc_recs)} recordings found on Xeno-canto")
            xc_count = xc_download(xc_recs, safe_dir)
        else:
            print(f"   XC:   0 recordings found")

        total = inat_count + xc_count
        print(f"   ✓ Total downloaded: {total}  (iNat={inat_count}, XC={xc_count})\n")

        summary_rows.append({
            "scientific_name": sci_name,
            "common_name":     common_name,
            "taxon_id":        taxon_id or "not_found",
            "inat_files":      inat_count,
            "xc_files":        xc_count,
            "total_files":     total,
            "status":          "ok" if total > 0 else "no_audio_found",
        })

    # ── Summary CSV ───────────────────────────────────────────────
    summary_path = OUTPUT_DIR / "dataset_summary_v2.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "scientific_name", "common_name", "taxon_id",
            "inat_files", "xc_files", "total_files", "status"
        ])
        writer.writeheader()
        writer.writerows(summary_rows)

    total_files   = sum(r["total_files"] for r in summary_rows)
    species_found = sum(1 for r in summary_rows if r["total_files"] > 0)
    empty         = [r["scientific_name"] for r in summary_rows if r["total_files"] == 0]

    print(f"\n{'='*65}")
    print(f"  ✅  Done!")
    print(f"  📦  {total_files} audio files across {species_found}/{len(SPECIES)} species")
    print(f"  📄  Summary → {summary_path}")
    if empty:
        print(f"\n  ⚠  Still 0 audio for {len(empty)} species:")
        for s in empty:
            print(f"      - {s}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()