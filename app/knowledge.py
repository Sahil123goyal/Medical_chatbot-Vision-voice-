from functools import lru_cache
from pinecone import Pinecone, ServerlessSpec
from groq import Groq
from sentence_transformers import SentenceTransformer
from app.config import settings

EMBED_DIM = 384  # all-MiniLM-L6-v2 outputs 384-dim vectors


@lru_cache()
def clients():
    if not settings.groq_api_key or not settings.pinecone_api_key:
        raise ValueError("Groq and Pinecone API keys must be set in the environment variables.")
    return Groq(api_key=settings.groq_api_key), Pinecone(api_key=settings.pinecone_api_key)


@lru_cache()
def embedder():
    return SentenceTransformer(settings.embedding_model)


def ensure_index():
    _, pc = clients()
    if not pc.has_index(settings.pinecone_index):
        pc.create_index(
            name=settings.pinecone_index,
            dimension=EMBED_DIM,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1'),
        )
    return pc.Index(settings.pinecone_index)


def embed(texts):
    return embedder().encode(texts, normalize_embeddings=True).tolist()


def search_kb(query, top_k=5):
    index = ensure_index()
    vector = embed([query])[0]
    result = index.query(vector=vector, top_k=top_k, include_metadata=True)
    return [
        {
            'page': m.metadata.get('page'),
            'text': m.metadata.get('text', ''),
            'score': round(m.score, 3),
            'source': 'Gale Encyclopedia (2002)',
        }
        for m in result.matches if m.score >= 0.30
    ]


def answer(query, top_k=5):
    groq, _ = clients()
    hits = search_kb(query, top_k)
    context = "\n\n".join(f"[p.{h['page']}] {h['text']}" for h in hits)
    resp = groq.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": "Answer using only the provided context. If it isn't there, say so."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content, hits