"""Download and index public Indian education policy/statistics PDFs."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import Request, urlopen

from indicrag.ingest import ingest_file, write_passages

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUTPUT = ROOT / "data" / "processed" / "passages.jsonl"

SOURCES = [
    {
        "filename": "nep_2020_english.pdf",
        "doc_id": "nep2020-full-en",
        "title": "National Education Policy 2020 · English full text",
        "language": "en",
        "url": "https://scert.delhi.gov.in/sites/default/files/SCERT/NEP%202020/Documents/nep_final_english_0.pdf",
    },
    {
        "filename": "nep_2020_hindi.pdf",
        "doc_id": "nep2020-full-hi",
        "title": "राष्ट्रीय शिक्षा नीति 2020 · हिंदी पूर्ण पाठ",
        "language": "hi",
        "url": "https://scert.delhi.gov.in/sites/default/files/SCERT/NEP%202020/Documents/nep_final_hindi_0.pdf",
    },
    {
        "filename": "udise_2024_25_nep.pdf",
        "doc_id": "udise-2024-25-nep",
        "title": "UDISE+ Report 2024–25 · NEP Structure",
        "language": "en",
        "url": "https://dashboard.udiseplus.gov.in/report2025/static/media/UDISE%2B2024_25_Booklet_nep.ea09e672a163f92d9cfe.pdf",
    },
    {
        "filename": "udise_2025_26_nep.pdf",
        "doc_id": "udise-2025-26-nep",
        "title": "UDISE+ Report 2025–26 · NEP Structure",
        "language": "en",
        "url": "https://dashboard.udiseplus.gov.in/report2026/static/media/UDISE%2B2025_26_Booklet_nep.94ceae1e8c2210549d21.pdf",
    },
]


def download(url: str, destination: Path, refresh: bool) -> None:
    if destination.exists() and not refresh:
        print(f"Using existing file: {destination.name}")
        return
    request = Request(url, headers={"User-Agent": "IndicRAG-QA research demo"})
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    total = 0
    try:
        with urlopen(request, timeout=60) as response, temporary.open("wb") as stream:
            while block := response.read(1024 * 1024):
                stream.write(block)
                total += len(block)
        if total < 1000:
            raise RuntimeError(f"Downloaded file is unexpectedly small ({total} bytes).")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Downloaded {destination.name} ({total / 1024 / 1024:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Download fresh copies of the source PDFs.")
    args = parser.parse_args()

    passages = []
    for source in SOURCES:
        path = RAW_DIR / source["filename"]
        download(source["url"], path, args.refresh)
        chunks = ingest_file(
            path,
            doc_id=source["doc_id"],
            title=source["title"],
            source_url=source["url"],
            language=source["language"],
            max_chars=900,
            overlap_chars=120,
        )
        passages.extend(chunks)
        print(f"Indexed {len(chunks):,} passages: {source['title']}")

    write_passages(passages, OUTPUT)
    print(f"Wrote {len(passages):,} total passages to {OUTPUT}")


if __name__ == "__main__":
    main()
