# ingest.py
import os, re, math, requests, psycopg2
from bs4 import BeautifulSoup

CONFLUENCE_URL = "https://confluence.lamoda.ru"
SPACE_KEY = "SITE"
CONFLUENCE_TOKEN = os.getenv("ATLASSIAN_API_TOKEN")
HEADERS = {"Authorization": f"Bearer {CONFLUENCE_TOKEN}"} if CONFLUENCE_TOKEN else {}

# адрес твоей embedding-модели (база), например: https://host:8000  или https://.../embeddings
EMBEDDING_BASE = (os.getenv("EMBEDDING_URL") or "").rstrip("/")
EMBEDDING_TOKEN = os.getenv("EMBEDDING_TOKEN")  # если требуется твоим сервисом

def fetch_pages():
    start = 0
    while True:
        url = f"{CONFLUENCE_URL}/rest/api/content?spaceKey={SPACE_KEY}&expand=body.storage&start={start}&limit=50"
        r = requests.get(url, headers=HEADERS); r.raise_for_status()
        data = r.json(); results = data.get("results", [])
        for p in results:
            yield {
                "page_id": p["id"],
                "title": p["title"],
                "html": p["body"]["storage"]["value"],
                "url": f"{CONFLUENCE_URL}/spaces/{SPACE_KEY}/pages/{p['id']}"
            }
        if len(results) < 50:
            break
        start += 50

def clean_text(text: str) -> str:
    """Убираем артефакты Confluence и приводим текст в порядок."""
    patterns_to_drop = [
        r"\{toc[^\}]*\}",              # {toc}, {toc:maxLevel=2}, ...
        r"\{expand[:\}]?.*?\}",        # {expand} (грубо)
        r"^\s*Table of Contents\s*$",
        r"^\s*Показать всё\s*$",
    ]
    for pat in patterns_to_drop:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    text = re.sub(r"[ \t]+", " ", text)         # множественные пробелы -> один
    text = re.sub(r"\n\s*\n+", "\n\n", text)    # много пустых строк -> одна
    return text.strip()

def to_chunks(text, max_chars=2500, overlap=200):
    i = 0
    while i < len(text):
        yield text[i:i + max_chars]
        i += max_chars - overlap

def _embed_post(url: str, payload: dict):
    headers = {"Content-Type": "application/json"}
    if EMBEDDING_TOKEN:
        headers["Authorization"] = f"Bearer {EMBEDDING_TOKEN}"
    r = requests.post(url, json=payload, headers=headers, timeout=60)
    return r

def _l2_norm_vecs(vecs):
    out = []
    for v in vecs:
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    return out

def embed_batch_passages(texts):
    """
    e5 ожидает префикс 'passage: ' для документов.
    Пытаемся сначала POST {base}/embed (TEI), затем — {base} или {base}/embeddings.
    Возвращаем список L2-нормированных векторов.
    """
    if not EMBEDDING_BASE:
        raise RuntimeError("EMBEDDING_URL не задан")

    prefixed = [f"passage: {t}" for t in texts]
    payload = {"inputs": prefixed}

    tried = []

    # 1) TEI-стиль: /embed
    url = f"{EMBEDDING_BASE}/embed"
    r = _embed_post(url, payload); tried.append((url, r.status_code))
    if r.status_code == 404:
        # 2) корневой: /
        url = EMBEDDING_BASE
        r = _embed_post(url, payload); tried.append((url, r.status_code))
        if r.status_code == 404:
            # 3) /embeddings
            url = f"{EMBEDDING_BASE}/embeddings"
            r = _embed_post(url, payload); tried.append((url, r.status_code))

    r.raise_for_status()
    js = r.json()

    # Поддержка разных форматов ответа
    if "embeddings" in js:
        embs = js["embeddings"]
    elif "data" in js and js["data"] and isinstance(js["data"][0], dict) and "embedding" in js["data"][0]:
        embs = [d["embedding"] for d in js["data"]]
    else:
        raise RuntimeError(f"Неожиданный ответ эмбеддинг-сервиса: keys={list(js.keys())}, tried={tried}")

    return _l2_norm_vecs(embs)

# ВАЖНО: для psycopg2 строка должна быть формата postgresql://user:pass@host:port/db
conn = psycopg2.connect(os.getenv("POSTGRES_URL")); conn.autocommit = True
cur = conn.cursor()

MIN_CHUNK_CHARS = 50  # отсекаем совсем короткие куски

for p in fetch_pages():
    # 1) HTML -> plain text
    plain = BeautifulSoup(p["html"], "html.parser").get_text(separator="\n")
    # 2) очистка
    plain = clean_text(plain)
    if not plain:
        continue

    # 3) разрезка на чанки
    chunks = [c for c in to_chunks(plain) if len(c.strip()) >= MIN_CHUNK_CHARS]
    if not chunks:
        continue

    # 4) эмбеддинги батчами и запись в БД
    for i in range(0, len(chunks), 32):
        batch = chunks[i:i + 32]
        embs = embed_batch_passages(batch)
        for j, (chunk, emb) in enumerate(zip(batch, embs)):
            cur.execute("""
                INSERT INTO confluence_chunks (page_id, page_title, space_key, url, chunk_idx, content, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (p["page_id"], p["title"], SPACE_KEY, p["url"], i + j, chunk, emb))

cur.close(); conn.close()
