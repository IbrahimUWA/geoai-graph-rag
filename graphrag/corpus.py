"""The document library: download the 32 ABARES reports and cut them into chunks with metadata.

Same corpus and chunking as the RAG lecture (geoai-rag-vector-database), so the two can be compared.
"""
import csv
import os
import re
from collections import Counter

import requests

from . import config

HEAD = {"User-Agent": "Mozilla/5.0 (GeoAI-Lab GraphRAG tutorial)"}
MONTHS = "January February March April May June July August September October November December".split()
BOILERPLATE = re.compile(
    r"Creative Commons (licence|Attribution)|Cataloguing data|Disclaimer|Professional independence|"
    r"acknowledge the continuous connection|Ownership of intellectual property|This report was prepared by|"
    r"Acknowledgements|ISSN \d{4}", re.I)


def manifest():
    path = os.path.join(config.PDFS, "manifest.csv")
    return {r["file"]: r for r in csv.DictReader(open(path, encoding="utf-8-sig"))}


def download():
    rows = manifest()
    for f, r in rows.items():
        out = os.path.join(config.PDFS, f)
        if os.path.exists(out) and open(out, "rb").read(4) == b"%PDF":
            continue
        resp = requests.get(r["url"], headers=HEAD, timeout=180)
        resp.raise_for_status()
        if not resp.content.startswith(b"%PDF"):
            print("not a PDF (the site may have moved it):", r["url"])
            continue
        open(out, "wb").write(resp.content)
        print(f"downloaded {f} ({len(resp.content) / 1e6:.1f} MB)")
    return len(rows)


def clean_page(text):
    text = text.replace("­", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    out = [l.strip() for l in text.splitlines()
           if l.strip() and not re.fullmatch(r"(page )?\d{1,3}( of \d+)?", l.strip().lower())]
    return re.sub(r"[ \t]+", " ", "\n".join(out)).strip()


def strip_repeated_lines(pages):
    cnt = Counter(l for p in pages for l in set(p.splitlines()) if len(l) < 120)
    common = {l for l, c in cnt.items() if len(pages) >= 5 and c / len(pages) > 0.4}
    return ["\n".join(l for l in p.splitlines() if l not in common) for p in pages]


def is_boilerplate(chunk):
    if len(re.findall(r"\.{6,}", chunk)) >= 3:
        return True
    return bool(BOILERPLATE.search(chunk)) and not re.search(r"\d+(\.\d+)? (million )?tonnes|per cent", chunk)


def chunk_page(text, tok):
    paras = [p for p in re.split(r"\n(?=[A-Z0-9•\-])", text) if p.strip()]
    chunks, cur, cur_len = [], [], 0
    size, overlap = config.CHUNK_TOKENS, config.CHUNK_OVERLAP
    for p in paras:
        n = len(tok.tokenize(p))
        if n > size:
            ids = tok(p, add_special_tokens=False)["input_ids"]
            chunks += [tok.decode(ids[i:i + size]) for i in range(0, len(ids), size - overlap)]
            continue
        if cur_len + n > size and cur:
            chunks.append("\n".join(cur))
            tail, tl = [], 0
            for q in reversed(cur):
                tl += len(tok.tokenize(q))
                tail.insert(0, q)
                if tl >= overlap:
                    break
            cur, cur_len = tail, tl
        cur.append(p)
        cur_len += n
    if cur:
        chunks.append("\n".join(cur))
    return [c.strip() for c in chunks if len(c.split()) >= 25 and not is_boilerplate(c)]


def header(title, date, page):
    pub = f", published {MONTHS[int(date[5:]) - 1]} {date[:4]}" if re.fullmatch(r"\d{4}-\d{2}", date or "") else ""
    return f"{title}{pub}, page {page}"


def build_chunks():
    import fitz
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(config.EMBEDDER)
    man = manifest()
    rows = []
    for f in sorted(x for x in os.listdir(config.PDFS) if x.lower().endswith(".pdf")):
        m = man.get(f, {})
        series = m.get("series") or f.split("_20")[0]
        title = m.get("title") or os.path.splitext(f)[0]
        pages = strip_repeated_lines([clean_page(p.get_text("text")) for p in fitz.open(os.path.join(config.PDFS, f))])
        for pno, text in enumerate(pages, start=1):
            for j, c in enumerate(chunk_page(text, tok)):
                rows.append({"id": f"{os.path.splitext(f)[0]}_p{pno}_{j}", "file": f, "title": title,
                             "series": series, "date": m.get("date", ""), "url": m.get("url", ""), "page": pno,
                             "access": config.access_for(series, f), "header": header(title, m.get("date", ""), pno),
                             "text": c})
    config.write_jsonl(config.CHUNKS, rows)
    return rows
