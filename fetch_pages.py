# ingest.py
import os, math, time, requests, psycopg2
from bs4 import BeautifulSoup
from requests.auth import HTTPBasicAuth
from openai import OpenAI

CONFLUENCE_URL = "https://your-domain.atlassian.net/wiki"
SPACE_KEY = "ENG"
auth = HTTPBasicAuth(os.getenv("ATLASSIAN_EMAIL"), os.getenv("ATLASSIAN_API_TOKEN"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def fetch_pages():
    start = 0
    while True:
        url = f"{CONFLUENCE_URL}/rest/api/content?spaceKey={SPACE_KEY}&expand=body.storage&start={start}&limit=50"
        r = requests.get(url, auth=auth); r.raise_for_status()
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
    # батчируйте для экономии
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [d.embedding for d in resp.data]

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
