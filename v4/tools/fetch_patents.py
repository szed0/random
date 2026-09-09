"""Build a real ORBIT-shaped patent export from Google Patents.

Stage 1 queries Google Patents once per (CPC area, two-year priority window),
so the corpus is a genuine hydrogen-technology landscape spread across time
rather than whatever relevance ranking returns first. Stage 2 fetches each
patent page and parses the seven columns the trt-pb Gems expect.

Columns produced (ORBIT default names):
  Title, Abstract, Publication numbers, Application dates,
  English description, Technology domains, Citing patents - Standardized
  publication number

Technology domains are CPC subclasses (four characters, e.g. H01M), newline
separated, at most four per patent - the same multi-valued shape ORBIT's
column has. Citing patents are the real forward citations from the page's
"Cited By" table. Descriptions are truncated so the file stays uploadable.

Google rate-limits: every request is paced, 503 backs off and retries, and
pages are cached on disk so a rerun only fetches what is missing.
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")
CACHE = "pat"
DESC_CHARS = 3000
MAX_DOMAINS = 4
PACE = 1.1                      # seconds between requests
BACKOFF = [15, 45, 90, 180]     # on 503

# Hydrogen technology, by CPC area. One coherent landscape, several domains.
AREAS = [
    ("fuel cells", "H01M8/04"),
    ("hydrogen production", "C01B3/00"),
    ("water electrolysis", "C25B1/04"),
    ("compressed gas storage", "F17C1/00"),
    ("hydride storage", "C01B6/00"),
    ("gas leak detection", "G01M3/00"),
]

_lock = threading.Lock()
_last = [0.0]


def get(url, tries=5):
    for attempt in range(tries):
        with _lock:
            wait = PACE - (time.time() - _last[0])
            if wait > 0:
                time.sleep(wait)
            _last[0] = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "en-US,en"})
            with urllib.request.urlopen(req, timeout=45) as r:
                return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 503) and attempt < tries - 1:
                nap = BACKOFF[min(attempt, len(BACKOFF) - 1)]
                print("  %d, backing off %ds" % (exc.code, nap), flush=True)
                time.sleep(nap)
                continue
            print("  fail %s (%s)" % (url[-32:], exc), flush=True)
            return None
        except Exception as exc:
            if attempt < tries - 1:
                time.sleep(5)
                continue
            print("  fail %s (%s)" % (url[-32:], exc), flush=True)
            return None


# --------------------------------------------------------------------------- #
# stage 1 - which patents
# --------------------------------------------------------------------------- #

def query(cpc, lo, hi, num):
    q = ("q=%s&before=priority:%d0101&after=priority:%d0101"
         "&country=US&status=GRANT&type=PATENT&language=ENGLISH&num=%d"
         % (urllib.parse.quote(cpc, safe=""), hi, lo, num))
    raw = get("https://patents.google.com/xhr/query?url=%s&exp=" % urllib.parse.quote(q, safe=""))
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    out = []
    for cluster in data.get("results", {}).get("cluster", []):
        for row in cluster.get("result", []):
            pn = row.get("patent", {}).get("publication_number")
            if pn:
                out.append(pn)
    return out


def collect_ids(windows, per_cell):
    ids = []
    for label, cpc in AREAS:
        got_area = 0
        for lo, hi in windows:
            got = query(cpc, lo, hi, per_cell)[:per_cell]
            got_area += len(got)
            ids.extend(got)
        print("%-24s %s  %d ids" % (label, cpc, got_area), flush=True)
    return list(dict.fromkeys(ids))


# --------------------------------------------------------------------------- #
# stage 2 - parse a patent page
# --------------------------------------------------------------------------- #

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def text_of(fragment):
    return _WS.sub(" ", html.unescape(_TAG.sub(" ", fragment))).strip()


def section(page, name):
    m = re.search(r'<section itemprop="%s" itemscope>(.*?)</section>' % name, page, re.S)
    return m.group(1) if m else ""


def parse(pn, page):
    abstract = re.sub(r"^Abstract\s*", "", text_of(section(page, "abstract")))
    if len(abstract) < 60:
        return None
    desc = re.sub(r"^Description\s*", "", text_of(section(page, "description")))[:DESC_CHARS]
    filed = re.findall(r'<meta name="DC.date" content="([\d-]+)" scheme="dateSubmitted"', page)
    prio = re.findall(r'itemprop="priorityDate"[^>]*>([\d-]{10})<', page)
    date = (filed or prio or [""])[0]
    if not date:
        return None
    tm = re.search(r'<meta name="DC.title" content="(.*?)"\s*/?>', page, re.S)
    codes = re.findall(r'<span itemprop="Code">([A-Z][\dA-Z/]*)</span>', page)
    subclasses = [c for c in dict.fromkeys(codes) if re.fullmatch(r"[A-Z]\d{2}[A-Z]", c)]
    if not subclasses:
        return None
    citing = []
    for block in re.findall(r'<tr itemprop="forwardReferences(?:Family|Orig)"[^>]*>(.*?)</tr>', page, re.S):
        m = re.search(r'<span itemprop="publicationNumber">([^<]+)</span>', block)
        if m:
            citing.append(m.group(1))
    return {
        "Title": text_of(tm.group(1)) if tm else "",
        "Abstract": abstract,
        "Publication numbers": pn,
        "Application dates": date.replace("-", "/"),
        "English description": desc,
        "Technology domains": "\n".join(subclasses[:MAX_DOMAINS]),
        "Citing patents - Standardized publication number": "\n".join(dict.fromkeys(citing)),
    }


def fetch(pn):
    path = os.path.join(CACHE, "%s.html" % pn)
    if os.path.exists(path) and os.path.getsize(path) > 20000:
        page = open(path, encoding="utf-8", errors="replace").read()
    else:
        page = get("https://patents.google.com/patent/%s/en" % pn)
        if not page:
            return None
        open(path, "w", encoding="utf-8").write(page)
    try:
        return parse(pn, page)
    except Exception as exc:
        print("  parse fail %s: %s" % (pn, exc), flush=True)
        return None


# --------------------------------------------------------------------------- #

def main():
    per_cell = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    os.makedirs(CACHE, exist_ok=True)
    windows = [(y, y + 2) for y in range(2005, 2023, 2)]
    print("querying %d CPC areas x %d windows x %d" % (len(AREAS), len(windows), per_cell), flush=True)
    ids = collect_ids(windows, per_cell)
    print("%d unique publication numbers\n" % len(ids), flush=True)

    rows = []
    for i, pn in enumerate(ids):
        row = fetch(pn)
        if row:
            rows.append(row)
        if (i + 1) % 20 == 0:
            print("  fetched %d/%d, kept %d" % (i + 1, len(ids), len(rows)), flush=True)

    frame = pd.DataFrame(rows).drop_duplicates(subset=["Publication numbers"]).reset_index(drop=True)
    frame.to_csv("patents.csv", index=False)

    years = frame["Application dates"].str[:4].astype(int)
    doms = pd.Series([d for v in frame["Technology domains"] for d in v.split("\n")])
    cites = frame["Citing patents - Standardized publication number"].map(
        lambda s: len(s.split("\n")) if isinstance(s, str) and s else 0)
    print("\nwrote patents.csv: %d rows, %d columns" % frame.shape, flush=True)
    print("  years       %d..%d" % (years.min(), years.max()))
    print("  per 2y      %s" % dict(years.groupby(years // 2 * 2).count()))
    print("  domains     %d distinct; top %s" % (doms.nunique(), dict(doms.value_counts().head(10))))
    print("  citations   total %d, median %d, max %d, zero %d"
          % (cites.sum(), cites.median(), cites.max(), (cites == 0).sum()))
    print("  abstract median %d chars, description median %d chars"
          % (frame["Abstract"].str.len().median(), frame["English description"].str.len().median()))


if __name__ == "__main__":
    main()
