#!/usr/bin/env python3
"""Fetch only license-qualified sources for EFTM-Open-R5 strict-open rebuild.

No use of earnings_latest.csv or any license-unknown mirror is permitted.
The script verifies Hugging Face dataset license metadata before download.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

LEDGER_REPO = "artefactory/ledger-market-sentiment"
SPEB_REPO = "speb/financial-data"
EXPECTED_LICENSE = "cc-by-4.0"

LEDGER_PATTERNS = [
    "README.md",
    "eps_surprise/data.parquet",
    "stock_prices/*.parquet",
]
SPEB_PATTERNS = [
    "README.md",
    "raw/stock_earning_calendar.parquet",
    "raw/stock_earnings_estimates.parquet",
    "raw/stock_earnings_history.parquet",
    "raw/stock_profile.parquet",
    "raw/stock_statement.parquet",
    "raw/stock_valuation_snapshot.parquet",
    "raw/stock_prices/**",
]

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def dataset_license(api: HfApi, repo_id: str) -> tuple[str|None,str|None]:
    info=api.dataset_info(repo_id)
    lic=None
    card=getattr(info, 'card_data', None)
    if card is not None:
        try: lic=card.get('license')
        except Exception: lic=getattr(card,'license',None)
    return (str(lic).lower() if lic else None, getattr(info,'sha',None))

def download_one(repo_id: str, target: Path, patterns: list[str], api: HfApi) -> dict:
    lic, sha = dataset_license(api, repo_id)
    if lic != EXPECTED_LICENSE:
        raise RuntimeError(f"License gate failed for {repo_id}: expected {EXPECTED_LICENSE}, got {lic!r}")
    target.mkdir(parents=True, exist_ok=True)
    local=snapshot_download(repo_id=repo_id, repo_type='dataset', revision=sha or 'main',
                            allow_patterns=patterns, local_dir=str(target))
    files=[]
    for p in sorted(Path(local).rglob('*')):
        if p.is_file() and '.cache' not in p.parts:
            files.append({'path':str(p.relative_to(target)), 'bytes':p.stat().st_size, 'sha256':sha256(p)})
    return {'repo_id':repo_id,'license':lic,'revision':sha,'local_dir':str(target),'files':files}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    args=ap.parse_args()
    src=args.root/'source_data'; src.mkdir(parents=True,exist_ok=True)
    api=HfApi()
    manifest={
      'policy':'strict-open-only',
      'excluded_sources':['earnings_latest.csv','Kaggle US historical stock prices with earnings data mirror (license Unknown)'],
      'sources':[]
    }
    manifest['sources'].append(download_one(LEDGER_REPO,src/'ledger',LEDGER_PATTERNS,api))
    manifest['sources'].append(download_one(SPEB_REPO,src/'speb',SPEB_PATTERNS,api))
    out=args.root/'SOURCE_LICENSE_MANIFEST.runtime.json'
    out.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'ok':True,'manifest':str(out)},indent=2))
if __name__=='__main__': main()
