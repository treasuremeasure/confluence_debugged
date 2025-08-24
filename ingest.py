# ingest.py
import os, re, requests, psycopg2
from urllib.parse import urlparse, parse_qs 
from bs4 import BeautifulSoup
import json
from dotenv import load_dotenv

load_dotenv()

CONFLUENCE_URL = os.getenv("CONFLUENCE_URL")
SPACE_KEY = os.getenv("SPACE_KEY")
ATLASSIAN_EMAIL = os.getenv("ATLASSIAN_EMAIL")  
ATLASSIAN_API_TOKEN = os.getenv("ATLASSIAN_API_TOKEN") 

API_BASE = f"{CONFLUENCE_URL}/wiki/rest/api"

EMBEDDING_URL = (os.getenv("EMBEDDING_URL"))

POSTGRES_URL = os.getenv("POSTGRES_URL")


def fetch_pages():  # CHANGED: реализация под Cloud /content/search + follow-up GET /content/{id}?expand=body.storage
    """
    Тянем страницы из одного SPACE по CQL:
      cql = "space=SPACE_KEY AND type=page"
    1) /wiki/rest/api/content/search?cql=...&limit=50[&cursor=...]
       Возвращает results[] + _links.next с cursor.
    2) Для каждой страницы отдельно тянем HTML: /wiki/rest/api/content/{id}?expand=body.storage
    Документация: 'Search content by CQL' (Cloud v1).  # см. ссылки в ответе
    """
    session = requests.Session()
    session.auth = (ATLASSIAN_EMAIL, ATLASSIAN_API_TOKEN)
    headers = {"Accept": "application/json"}

    cql = f'space="{SPACE_KEY}" AND type=page'
    limit = 50
    cursor = None

    while True:
        params = {"cql": cql, "limit": limit}
        if cursor:
            params["cursor"] = cursor
        url = f"{API_BASE}/content/search"
        r = session.get(url, params=params, headers=headers)
        try:
            print("Cтраницы успешно получены:", r.status_code)
        except:
            raise requests.HTTPError(f"HTTP {r.status_code}: {r.reason}")
        data = r.json()

        for item in data.get("results", []):
            page_id = item.get("id")
            title = item.get("title") or ""
            # второй запрос: тянем body.storage для конкретной страницы
            r2 = session.get(f"{API_BASE}/content/{page_id}",
                             params={"expand": "body.storage"}, headers=headers)
            r2.raise_for_status()
            js2 = r2.json()
            html = (js2.get("body", {}).get("storage", {}) or {}).get("value", "")  # HTML тела страницы

            yield {
                "page_id": page_id,
                "title": title,
                "html": html,
                # Cloud формирует _links для страницы; соберём URL на страницу
                "url": f"{CONFLUENCE_URL}/spaces/{SPACE_KEY}/pages/{page_id}"
            }

        # курсорная пагинация: берём cursor из _links.next
        next_rel = (data.get("_links") or {}).get("next")
        if not next_rel:
            break
        # из next_rel вытаскиваем cursor, чтобы снова ходить на один и тот же endpoint
        q = parse_qs(urlparse(next_rel).query)
        cursor = (q.get("cursor") or [None])[0]
        if not cursor:
            break

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
    # NOTE: простой символьный чанкёр; если нужна «аккуратность», режь по абзацам/точкам.
    i = 0
    step = max_chars - overlap
    if step <= 0:  # FIX: защита от неправильных параметров
        step = max_chars
    while i < len(text):
        yield text[i:i + max_chars]
        i += step

def _embed_post(url: str, payload: dict):
    """POST к сервису эмбеддингов. Токен не используем — при необходимости добавь headers."""
    if not url:
        raise ValueError("EMBEDDING_URL не задан")
    headers = {"Content-Type": "application/json"}
    r = requests.post(url, json=payload, headers=headers, timeout=60)
    try:
        print("Ответ успешно получен (embed_post)", r.status_code)
    except:
        raise requests.HTTPError(f"HTTP {r.status_code}: {r.reason}")
    return r

def _is_vector(x):
    return isinstance(x, list) and len(x) > 0 and all(isinstance(v, (int, float)) for v in x)

def _is_list_of_vectors(x):
    return (isinstance(x, list) and len(x) > 0
            and all(isinstance(v, list) and len(v) > 0 and all(isinstance(t, (int, float)) for t in v) for v in x))

def _unwrap_once(x):
    # строка JSON → объект
    if isinstance(x, str):
        try:
            return json.loads(x)
        except json.JSONDecodeError:
            # если это вовсе не JSON — оставим как есть
            return x
    # некоторые рантаймы отдают ещё один уровень-обёртку длины 1
    if isinstance(x, list) and len(x) == 1 and isinstance(x[0], (list, str)):
        return _unwrap_once(x[0])
    return x

def _parse_embeddings_triton(js: dict, expected_n: int):
    outs = js.get("outputs") or js.get("data") or js.get("predictions")
    if not outs:
        raise ValueError(f"В ответе нет outputs/data/predictions. Ключи: {list(js.keys())}")

    out = outs[0] if isinstance(outs, list) else outs
    dtype = (out.get("datatype") or "").upper()
    params = out.get("parameters") or {}
    ctype = (params.get("content_type") or "").lower()
    shape = out.get("shape")
    data  = out.get("data")

    # FP32 [N, D]
    if dtype in {"FP32", "FLOAT32"} and isinstance(shape, list) and len(shape) == 2:
        n, d = shape
        flat = data if isinstance(data, list) else []
        embs = [flat[i*d:(i+1)*d] for i in range(n)] if d and flat else []
        return embs  # БЕЗ проверок количества

    # BYTES + hg_jsonlist
    if dtype == "BYTES" and ctype == "hg_jsonlist":
        vectors = []
        for item in (data if isinstance(data, list) else []):
            item = _unwrap_once(item)          # [[[...]]] -> [[...]] -> [...]
            # !!! ключевой момент: не использовать extend на плоском векторе
            if _is_vector(item):               # один вектор
                vectors.append(item)
            elif _is_list_of_vectors(item):    # список векторов
                vectors.extend(item)
            else:
                item2 = _unwrap_once(item)
                if _is_vector(item2):
                    vectors.append(item2)
                elif _is_list_of_vectors(item2):
                    vectors.extend(item2)
                else:
                    # если вообще пришёл плоский список чисел — считаем это одним вектором
                    if isinstance(item2, list) and all(isinstance(x, (int, float)) for x in item2):
                        vectors.append(item2)
                    else:
                        print(f"[embed] warning: нераспознанный элемент: {str(item)[:160]!r}")
        return vectors

    print(f"[embed] warning: неожиданный формат выхода: dtype={dtype}, content_type={ctype}, shape={shape}; пытаюсь вернуть как есть")
    # Последняя попытка нормализации
    if _is_vector(data):
        return [data]
    if _is_list_of_vectors(data):
        return data
    return []



def embed_batch_passages(texts: list[str]):
    """
    Отправляем чанки в сервис эмбеддингов.
    Формат запроса (Triton-like):
    {
      "inputs": [
        {
          "name": "args",
          "shape": [-1],
          "datatype": "BYTES",
          "data": ["passage: ...", "passage: ..."]
        }
      ]
    }
    """
    if not EMBEDDING_URL:
        raise RuntimeError("EMBEDDING_URL не задан")

    # e5-модели ждут префикс passage: / query:
    prefixed = [f"passage: {t}" for t in texts]

    payload = {
        "inputs": [
            {
                "name": "args",
                "shape": [-1],
                "datatype": "BYTES",
                "data": prefixed
            }
        ]
    }

    r = _embed_post(EMBEDDING_URL, payload)
    print("status:", r.status_code)
    print("content-type:", r.headers.get("content-type"))
    print("first 300 bytes:", r.text[:300])
    js = r.json()  # сейчас приходит application/json
    embs = _parse_embeddings_triton(js, expected_n=len(texts))

    # НОРМАЛИЗАЦИЯ: гарантируем list[list[float]]
    if embs and all(isinstance(x, (int, float)) for x in embs):
        # пришёл плоский список чисел → один вектор
        embs = [embs]
    elif embs and isinstance(embs, list) and embs and isinstance(embs[0], list):
        pass  # всё ок
    else:
        print("[embed] warning: парсер вернул неожиданный формат, принял как пусто")
        embs = []

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

