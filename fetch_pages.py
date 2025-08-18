# ingest.py
import os, requests, psycopg2
from bs4 import BeautifulSoup

CONFLUENCE_URL = "https://confluence.lamoda.ru"
SPACE_KEY = "SITE"
CONFLUENCE_TOKEN = os.getenv("ATLASSIAN_API_TOKEN")
HEADERS = {"Authorization": f"Bearer {CONFLUENCE_TOKEN}"}

# адрес твоей embedding модели
EMBEDDING_URL = os.getenv("EMBEDDING_URL")

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
        if len(results) < 50: break
        start += 50

def to_chunks(text, max_chars=2500, overlap=200):
    i = 0
    while i < len(text):
        yield text[i:i+max_chars]
        i += max_chars - overlap

def embed_batch(texts):
    resp = requests.post(EMBEDDING_URL, json={"inputs": texts})
    resp.raise_for_status()
    return resp.json()["embeddings"]

conn = psycopg2.connect(os.getenv("POSTGRES_URL")); conn.autocommit = True
cur = conn.cursor()

for p in fetch_pages():
    plain = BeautifulSoup(p["html"], "html.parser").get_text(separator="\n")
    chunks = list(to_chunks(plain))
    for i in range(0, len(chunks), 32):
        batch = chunks[i:i+32]
        embs = embed_batch(batch)
        for j, (chunk, emb) in enumerate(zip(batch, embs)):
            cur.execute("""
              INSERT INTO confluence_chunks(page_id,page_title,space_key,url,chunk_idx,content,embedding)
              VALUES (%s,%s,%s,%s,%s,%s,%s)
              ON CONFLICT DO NOTHING
            """, (p["page_id"], p["title"], SPACE_KEY, p["url"], i+j, chunk, emb))
cur.close(); conn.close()
