"""
scripts/fetch_euvsdisinfo.py — build data/euvsdisinfo/cases.jsonl (the narrative base).

Sources:
  1. Mendeley dump (CC BY 4.0): 14,497 cases 2015-01 .. 2022-11, downloaded to mendeley_2015_2022.csv
  2. euvsdisinfo.eu itself for everything newer (Cloudflare needs browser TLS → curl_cffi)

Steps (each resumable, run in order):
  uv run --group data python scripts/fetch_euvsdisinfo.py listing [per_page]   # newest→oldest until Mendeley range
  uv run --group data python scripts/fetch_euvsdisinfo.py reports    # one page per new case, cached
  uv run --group data python scripts/fetch_euvsdisinfo.py merge      # → cases.jsonl

Case record (cases.jsonl):
  id, url, date (ISO), title, summary, disproof, outlets [names], sources [original URLs],
  telegram_channels [usernames], countries [], tags [], origin ("mendeley" | "site")
"""

from __future__ import annotations

import base64
import csv
import html
import json
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path

DATA = Path(__file__).parent.parent / "data" / "euvsdisinfo"
MENDELEY_CSV = DATA / "mendeley_2015_2022.csv"
MENDELEY_URL = (
    "https://data.mendeley.com/public-files/datasets/yhdtkszvgp/files/"
    "481ecd12-8a6e-4c98-8bf5-3d8cc58703fa/file_downloaded"
)
LISTING = DATA / "listing_new.jsonl"
REPORTS = DATA / "reports"
CASES = DATA / "cases.jsonl"
BASE = "https://euvsdisinfo.eu"
MENDELEY_LAST_DATE = date(2022, 11, 22)
DELAY = 0.5  # seconds between requests per worker — be polite


def _session():
    from curl_cffi import requests

    return requests.Session(impersonate="chrome", timeout=40)


def _get(sess, url: str, retries: int = 4) -> str:
    for attempt in range(retries):
        r = sess.get(url)
        if r.status_code == 200:
            return r.text
        wait = 5 * (attempt + 1)
        print(f"  HTTP {r.status_code} for {url} — retry in {wait}s", file=sys.stderr)
        time.sleep(wait)
    raise RuntimeError(f"giving up on {url}")


def _clean(fragment: str) -> str:
    text = re.sub(r"<[^>]+>", " ", fragment)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def _parse_date(s: str) -> str:
    s = s.strip()
    for fmt in ("%d.%m.%Y", "%B %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    return s


# ── 1. listing ───────────────────────────────────────────────────────────────
# The HTML listing is capped at 600 results, but the theme's "load more" AJAX endpoint
# (cms/wp-admin/admin-ajax.php, action=loadmore) pages through the whole database.

AJAX = f"{BASE}/cms/wp-admin/admin-ajax.php"
_ROW_RE = re.compile(
    r'disinfo-db-date"[^>]*>\s*([^<]+?)\s*</td>.*?href="([^"]+/report/[^"]+)"\s*>\s*(.*?)\s*</a>'
    r".*?disinfo-outlets-list'>\s*(.*?)\s*</div>.*?cell-country\"[^>]*>\s*([^<]*?)\s*</td>",
    re.DOTALL,
)


def fetch_listing_page(sess, page: int, per_page: int = 10) -> list[dict]:
    r = sess.post(
        AJAX,
        data={
            "action": "loadmore",
            "query": str(per_page),
            "page": page,
            "textQuery": "",
            "keywords": "",
            "date": "",
            "countries": "",
            "language": "",
            "outlets": "",
        },
    )
    if r.status_code != 200:
        raise RuntimeError(f"AJAX page {page}: HTTP {r.status_code}")
    out = []
    for d, url, title, outlets, country in _ROW_RE.findall(r.text):
        out.append(
            {
                "id": url.strip("/").split("/")[-1],
                "url": url,
                "date": _parse_date(d),
                "title": _clean(title),
                "outlets": [o.strip() for o in _clean(outlets).split(",") if o.strip()],
                "countries": [c.strip() for c in _clean(country).split(",") if c.strip()],
            }
        )
    return out


def cmd_listing() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    if not MENDELEY_CSV.exists():
        print("Mendeley CSV missing — downloading")
        MENDELEY_CSV.write_bytes(_session().get(MENDELEY_URL).content)
    known = {row["Links"].strip("/").split("/")[-1] for row in _read_mendeley()}
    seen: dict[str, dict] = {}
    if LISTING.exists():
        for line in LISTING.read_text().splitlines():
            rec = json.loads(line)
            seen[rec["id"]] = rec
    sess = _session()
    sess.get(f"{BASE}/disinformation-cases/")  # cookies
    per_page = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    page, stop, empty = 1, False, 0
    with LISTING.open("a") as out:
        while not stop:
            items = fetch_listing_page(sess, page, per_page)
            if not items:
                empty += 1
                if empty >= 3:
                    break
                time.sleep(5)
                continue
            empty = 0
            new = 0
            for rec in items:
                if date.fromisoformat(rec["date"]) < MENDELEY_LAST_DATE and rec["id"] in known:
                    stop = True
                if rec["id"] in known or rec["id"] in seen:
                    continue
                seen[rec["id"]] = rec
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                new += 1
            if page % 10 == 0 or stop:
                print(
                    f"page {page}: {len(items)} items, {new} new, last date {items[-1]['date']}, total new {len(seen)}"
                )
            page += 1
            time.sleep(DELAY)
    print(f"listing done: {len(seen)} new cases in {LISTING}")


# ── 2. reports ───────────────────────────────────────────────────────────────


def _block(page: str, cls: str) -> str:
    m = re.search(rf'class="{cls}".*?class="b-text"[^>]*>(.*?)</div>', page, re.DOTALL)
    return _clean(m.group(1)) if m else ""


def _li(page: str, label: str) -> str:
    m = re.search(rf"<li>\s*{label}:?(.*?)</li>", page, re.DOTALL)
    return m.group(1) if m else ""


def _decode_seo(href: str) -> str:
    try:
        return base64.b64decode(href + "=" * (-len(href) % 4)).decode()
    except Exception:
        return ""


def parse_report(page: str) -> dict:
    outlets_html = _li(page, "Outlet")
    names = [_clean(n) for n in re.findall(r'rel="nofollow">([^<]+)</a>\s*\(', outlets_html)]
    originals = [_decode_seo(h) for h in re.findall(r'data-seo-href="([^"]+)"[^>]*>original<', outlets_html)]
    tg = sorted(
        {m.group(1) for u in originals if (m := re.match(r"https?://t\.me/(?:s/)?([A-Za-z0-9_]+)", u))}
    )
    tags_html = re.search(r'b-report__keywords">.*?Tags:(.*?)</div>', page, re.DOTALL)
    return {
        "summary": _block(page, "b-report__summary"),
        "disproof": _block(page, "b-report__response"),
        "outlets": names,
        "sources": [u for u in originals if u],
        "telegram_channels": tg,
        "countries": [
            c.strip() for c in _clean(_li(page, "Countries / regions discussed")).split(",") if c.strip()
        ],
        "tags": [_clean(t) for t in re.findall(r"<span>([^<]+)</span>", tags_html.group(1))]
        if tags_html
        else [],
        "date": _parse_date(_clean(_li(page, "Date of publication"))),
    }


def _fetch_one(rec: dict, sess) -> tuple[str, dict | None]:
    try:
        parsed = parse_report(_get(sess, rec["url"]))
    except Exception as exc:  # keep going; rerun picks up the gaps
        print(f"  FAIL {rec['id']}: {exc}", file=sys.stderr)
        return rec["id"], None
    time.sleep(DELAY)
    return rec["id"], {**rec, **parsed}


def cmd_reports() -> None:
    """Fetch every listed case page (N threads, each with its own session), cache one JSON per case."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    REPORTS.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(line) for line in LISTING.read_text().splitlines()]
    todo = [r for r in recs if not (REPORTS / f"{r['id']}.json").exists()]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    print(f"{len(recs)} listed, {len(todo)} to fetch, {workers} workers", flush=True)
    local = threading.local()

    def work(rec):
        if not hasattr(local, "sess"):
            local.sess = _session()
        return _fetch_one(rec, local.sess)

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for case_id, data in pool.map(work, todo):
            if data is not None:
                (REPORTS / f"{case_id}.json").write_text(json.dumps(data, ensure_ascii=False))
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(todo)}", flush=True)
    print("reports done", flush=True)


# ── 3. merge ─────────────────────────────────────────────────────────────────


def _read_mendeley() -> list[dict]:
    csv.field_size_limit(10**9)
    with MENDELEY_CSV.open(encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def cmd_merge() -> None:
    n = 0
    with CASES.open("w") as out:
        for row in _read_mendeley():
            path = row["Links"].strip()
            outlets = [o.strip() for o in row["Outlets"].split(",") if o.strip()]
            rec = {
                "id": path.strip("/").split("/")[-1],
                "url": BASE + path,
                "date": _parse_date(row["Date"]),
                "title": row["Title"].strip(),
                "summary": row["Disinformation"].strip(),
                "disproof": row["Information"].strip(),
                "outlets": outlets,
                "sources": [],
                "telegram_channels": [],
                "countries": [c.strip() for c in row["Country"].split(",") if c.strip()],
                "tags": [],
                "origin": "mendeley",
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
        listed = [json.loads(line) for line in LISTING.read_text().splitlines()] if LISTING.exists() else []
        missing = 0
        for rec in listed:
            p = REPORTS / f"{rec['id']}.json"
            if p.exists():
                rec = {**rec, **json.loads(p.read_text()), "origin": "site"}
            else:  # page not fetched (yet): keep the title-only record so the case is still searchable
                rec = {
                    **rec,
                    "summary": "",
                    "disproof": "",
                    "sources": [],
                    "telegram_channels": [],
                    "tags": [],
                    "origin": "listing",
                }
                missing += 1
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
        if missing:
            print(f"note: {missing} cases have title only (run `reports` to fetch their pages)")
    print(f"merged {n} cases → {CASES}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    {"listing": cmd_listing, "reports": cmd_reports, "merge": cmd_merge}.get(cmd, lambda: print(__doc__))()
