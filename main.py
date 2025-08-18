# main.py
import os, psycopg2, requests
import chainlit as cl

# адреса сервисов
EMBEDDING_URL = os.getenv("EMBEDDING_URL")
LLM_URL = os.getenv("LLM_URL")
LLM_MODEL = os.getenv("LLM_MODEL", "TheBloke/deepseek-llm-67b-chat-AWQ")

def embed_text(text: str):
    """Получаем эмбеддинг текста с локальной модели"""
    resp = requests.post(EMBEDDING_URL, json={"inputs": [text]})
    resp.raise_for_status()
    return resp.json()["embeddings"][0]

def search_chunks(question: str, top_k: int = 5, metric: str = "cosine"):
    """Ищем ближайшие чанки в Postgres по векторному сходству"""
    conn = psycopg2.connect(os.getenv("POSTGRES_URL"))
    cur = conn.cursor()
    emb = embed_text(question)
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
    """Стримим ответ из локальной LLM через vLLM (OpenAI совместимый API)"""
    with requests.post(
        LLM_URL,
        json={
            "model": LLM_MODEL,
            "messages": messages,
            "stream": True,
        },
        stream=True,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[len("data: "):]
                    if data.strip() == "[DONE]":
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
    context = "\n\n".join(
        [f"[{h['title']}]({h['url']})\n{h['content']}" for h in hits]
    )

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
