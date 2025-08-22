# ingest.py
import os, re, requests, psycopg2
from bs4 import BeautifulSoup

CONFLUENCE_URL = "https://confluence.lamoda.ru"
SPACE_KEY = "SITE"
CONFLUENCE_TOKEN = os.getenv("ATLASSIAN_API_TOKEN")
HEADERS = {"Authorization": f"Bearer {CONFLUENCE_TOKEN}"} if CONFLUENCE_TOKEN else {}

EMBEDDING_URL = (os.getenv("EMBEDDING_URL"))

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
    """Простой POST без авторизации (токен не нужен). Возвращает requests.Response."""
    if not url:
        raise RuntimeError("EMBEDDING_URL не задан")
    headers = {"Content-Type": "application/json"}
    r = requests.post(url, json=payload, headers=headers, timeout=60)
    r.raise_for_status()
    return r

def embed_batch_passages(texts):
   
    if not EMBEDDING_URL:
        raise RuntimeError("EMBEDDING_URL не задан")

    prefixed = [f"passage: {t}" for t in texts]
    payload = {"input_texts": prefixed}

    r = _embed_post(EMBEDDING_URL, payload)
    js = r.json()

    # Разбор двух популярных форматов
    if isinstance(js, dict) and "embeddings" in js:
        embs = js["embeddings"]
    elif isinstance(js, dict) and "data" in js:
        embs = [item["embedding"] for item in js["data"]]
    else:
        raise ValueError(f"Неожиданный формат ответа от эмбеддинга: {js}")

    # Быстрая валидация длины
    if len(embs) != len(texts):
        raise ValueError(f"Кол-во эмбеддингов ({len(embs)}) != кол-ву текстов ({len(texts)})")

    return embs


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

