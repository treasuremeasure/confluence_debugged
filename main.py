# main.py
import os, math, psycopg2, requests
import chainlit as cl

EMBEDDING_BASE = os.getenv("EMBEDDING_URL", "").rstrip("/")
LLM_URL = os.getenv("LLM_URL", "http://127.0.0.1:8001/v1/chat/completions")
LLM_MODEL = os.getenv("LLM_MODEL", "TheBloke/deepseek-llm-67b-chat-AWQ")

def l2_normalize(vec):
    s = math.sqrt(sum(x*x for x in vec)) or 1.0
    return [x/s for x in vec]

def embed_query(text: str):
    # e5 ожидает префикс "query: "
    payload = {"inputs": [f"query: {text}"]}
    # пробуем TEI /embed
    url = EMBEDDING_BASE + "/embed" if EMBEDDING_BASE else ""
    r = requests.post(url or EMBEDDING_BASE, json=payload) if EMBEDDING_BASE else None

    if r is None or r.status_code == 404:
        # fallback на корневой /embeddings-стиль
        r = requests.post(EMBEDDING_BASE or "", json=payload)
    r.raise_for_status()
    js = r.json()
    emb = None
    if "embeddings" in js:
        emb = js["embeddings"][0]
    elif "data" in js and js["data"]:
        emb = js["data"][0].get("embedding")
    if emb is None:
        raise RuntimeError(f"Embedding API returned unexpected payload: {js}")
    return l2_normalize(emb)  # (опц.) L2-норм

def search_chunks(question: str, top_k: int = 5, metric: str = "cosine"):
    conn = psycopg2.connect(os.getenv("POSTGRES_URL"))
    cur = conn.cursor()
    emb = embed_query(question)
    op = "<=>" if metric == "cosine" else "<->"
    cur.execute(f"""
      SELECT page_title, url, content
      FROM confluence_chunks
      ORDER BY embedding {op} %s
      LIMIT %s
    """, (emb, top_k))
    rows = cur.fetchall()
    cur.close(); conn.close()
    return [{"title": r[0], "url": r[1], "content": r[2]} for r in rows]

def stream_llm(messages):
    with requests.post(
        LLM_URL,
        json={"model": LLM_MODEL, "messages": messages, "stream": True},
        stream=True,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            s = line.decode("utf-8")
            if not s.startswith("data: "):
                continue
            data = s[6:].strip()
            if data == "[DONE]":
                break
            try:
                token = requests.utils.json.loads(data)["choices"][0]["delta"].get("content")
                if token:
                    yield token
            except Exception:
                continue

@cl.on_message
async def on_message(message: cl.Message):
    question = message.content.strip()
    hits = search_chunks(question)
    context = "\n\n".join([f"[{h['title']}]({h['url']})\n{h['content']}" for h in hits])

    sys = "Ты корпоративный помощник. Отвечай кратко, добавляй ссылки на источники."
    user = f"Вопрос: {question}\n\nКонтекст:\n{context}\n\nОтветь со ссылками на релевантные страницы."

    msg = cl.Message(content="")
    await msg.send()

    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": user},
    ]

    for token in stream_llm(messages):
        await msg.stream_token(token)

    await msg.update()
