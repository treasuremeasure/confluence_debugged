# main.py (Chainlit)
import os, psycopg2
import chainlit as cl

def search_chunks(question: str, top_k: int = 5, metric: str = "cosine"):
    conn = psycopg2.connect(os.getenv("POSTGRES_URL"))
    cur = conn.cursor()
    emb = client.embeddings.create(model="text-embedding-3-small", input=question).data[0].embedding
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

@cl.on_message
async def on_message(message: cl.Message):
    question = message.content.strip()
    hits = search_chunks(question)
    context = "\n\n".join([f"[{h['title']}]({h['url']})\n{h['content']}" for h in hits])

    sys = "Ты корпоративный помощник. Отвечай кратко, добавляй ссылки на источники."
    user = f"Вопрос: {question}\n\nКонтекст:\n{context}\n\nОтветь со ссылками на релевантные страницы."

    # создаём пустое сообщение и стримим в него токены
    msg = cl.Message(content="")
    await msg.send()

    with client.chat.completions.with_streaming_response.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":sys},{"role":"user","content":user}],
        stream=True
    ) as resp:
        async for event in resp:
            token = getattr(event, "delta", None)
            if token:
                await msg.stream_token(token)

    await msg.update()
