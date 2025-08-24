# main.py
import os, psycopg2, requests, json
import chainlit as cl
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_URL = os.getenv("EMBEDDING_URL")
LLM_URL = os.getenv("LLM_URL")
LLM_MODEL = os.getenv("LLM_MODEL").rstrip("/")
EXPECTED_DIM = int(os.getenv("EMBEDDING_DIM", "0"))

def _is_vector(x):
    return isinstance(x, list) and x and all(isinstance(v, (int, float)) for v in x)

def _is_list_of_vectors(x):
    return (isinstance(x, list) and x and
            all(isinstance(v, list) and v and all(isinstance(t, (int, float)) for t in v) for v in x))

def _unwrap_once(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except json.JSONDecodeError:
            return x
    if isinstance(x, list) and len(x) == 1 and isinstance(x[0], (list, str)):
        return _unwrap_once(x[0])
    return x

def _parse_embeddings_response(js):
    """
    Возвращает list[list[float]].
    Поддерживает:
      - MLServer/Triton BYTES + parameters.content_type=='hg_jsonlist'
      - Классический FP32 тензор [N, D] с плоским data
      - Упрощённые ответы вида {"embeddings":[...]} / {"data":[{"embedding":[...]}]}
    """
    # Простые варианты (OpenAI-like / свои прокси)
    if "embeddings" in js and isinstance(js["embeddings"], list):
        embs = js["embeddings"]
        return embs if _is_list_of_vectors(embs) else [embs]
    if "data" in js and isinstance(js["data"], list) and js["data"]:
        first = js["data"][0]
        if isinstance(first, dict) and "embedding" in first:
            return [first["embedding"]]

    # Triton/MLServer
    outs = js.get("outputs") or js.get("data") or js.get("predictions")
    if outs:
        out = outs[0] if isinstance(outs, list) else outs
        dtype = (out.get("datatype") or "").upper()
        params = out.get("parameters") or {}
        ctype = (params.get("content_type") or "").lower()
        shape = out.get("shape")
        data  = out.get("data")

        # FP32 с плоским data
        if dtype in {"FP32", "FLOAT32"} and isinstance(shape, list) and len(shape) == 2:
            n, d = shape
            flat = data if isinstance(data, list) else []
            embs = [flat[i*d:(i+1)*d] for i in range(n)] if d and flat else []
            return embs

        # BYTES + hg_jsonlist (твой случай)
        if dtype == "BYTES" and ctype == "hg_jsonlist":
            vectors = []
            for item in (data if isinstance(data, list) else []):
                item = _unwrap_once(item)  # [[[...]]] -> [[...]] -> [...]
                if _is_vector(item):
                    vectors.append(item)
                elif _is_list_of_vectors(item):
                    vectors.extend(item)
                else:
                    item2 = _unwrap_once(item)
                    if _is_vector(item2):
                        vectors.append(item2)
                    elif _is_list_of_vectors(item2):
                        vectors.extend(item2)
                    elif isinstance(item2, list) and all(isinstance(x, (int, float)) for x in item2):
                        vectors.append(item2)
            # Если вдруг вернулся "плоский" список чисел (ошибка парсинга) — обернём как один вектор
            if vectors and all(isinstance(x, (int, float)) for x in vectors):
                return [vectors]
            return vectors

    raise RuntimeError(f"Embedding API returned unexpected payload shape: keys={list(js.keys())}")

def _fit_dim(vec, target):
    if not target or target <= 0:
        # без подгонки — вернуть как есть
        return vec
    n = len(vec)
    if n == target:
        return vec
    if n > target:
        return vec[:target]          # мягкая обрезка
    return vec + [0.0] * (target - n) # дополнение нулями


def embed_query(text: str):
    if not EMBEDDING_URL:
        raise RuntimeError("EMBEDDING_URL не задан")

    # Сначала пробуем Triton/MLServer формат (как в ingest.py)
    payload = {
        "inputs": [
            {
                "name": "args",
                "shape": [-1],
                "datatype": "BYTES",
                "data": [f"query: {text}"]
            }
        ]
    }
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    r = requests.post(EMBEDDING_URL, json=payload, headers=headers, timeout=(5, 60))
    try:
        print("Ответ успешно получен (embed_post)", r.status_code)
    except:
        raise requests.HTTPError(f"HTTP {r.status_code}: {r.reason}")

    # Разбор ответа
    js = r.json()
    embs = _parse_embeddings_response(js)

    # Нормализуем к одному вектору (мы отправляли 1 вход)
    if not embs:
        raise RuntimeError(f"Embedding API returned empty embeddings for query: {text!r}")
    emb = embs[0] if _is_list_of_vectors(embs) else (embs if _is_vector(embs) else None)
    if emb is None:
        raise RuntimeError(f"Embedding API returned unexpected embeddings: {type(embs)}")

    # Мягкая подгонка под размерность колонки (если задана EMBEDDING_DIM)
    emb = _fit_dim(emb, EXPECTED_DIM)

    return emb  # list[float]

def search_chunks(question: str, top_k: int = 5, metric: str = "cosine"):
    conn = psycopg2.connect(os.getenv("POSTGRES_URL"))
    cur = conn.cursor()
    emb = embed_query(question)
    op = "<=>" if metric == "cosine" else "<->"
    cur.execute(f"""
      SELECT page_title, url, content
      FROM confluence_chunks
      ORDER BY embedding {op} %s::vector
      LIMIT %s
    """, (emb, top_k))
    rows = cur.fetchall()
    cur.close(); conn.close()
    return [{"title": r[0], "url": r[1], "content": r[2]} for r in rows]

def stream_llm(messages):
    url = f"{LLM_URL}/v1/chat/completions"
    with requests.post(
        url,
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

    print(EMBEDDING_URL)

    msg = cl.Message(content="")
    await msg.send()

    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": user},
    ]

    for token in stream_llm(messages):
        await msg.stream_token(token)

    await msg.update()
