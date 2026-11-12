# Local embedding backend setup

This patch adds an optional CPU Sentence Transformers backend. Gemini remains available for answer generation and as the default embedding backend.

## Install

From the repository root, with the project's virtual environment activated:

```bash
python -m pip install -e '.[local-embeddings]'
```

The first use downloads `sentence-transformers/all-MiniLM-L6-v2` from Hugging Face if it is not already cached. Embedding inference after that runs locally on CPU.

## Select local embeddings

Set this in `.env` (do not put your API key in any shared file):

```dotenv
FINAGENT_EMBEDDING_BACKEND=sentence-transformers
FINAGENT_LOCAL_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

Or set it for one terminal session:

```bash
export FINAGENT_EMBEDDING_BACKEND=sentence-transformers
export FINAGENT_LOCAL_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

Run retrieval evaluation only after installation:

```bash
finagent eval-retrieval --variant full --retriever dense --split dev --limit 20 --out runs/retrieval-full-dense-local.jsonl
finagent eval-retrieval --variant full --retriever hybrid --split dev --limit 20 --out runs/retrieval-full-hybrid-local.jsonl
```

These commands should not call Gemini for embeddings. The index key includes the backend/model name, so the local model rebuilds its own index rather than reusing Gemini vectors. The evaluator may still need to download benchmark/PDF data if those files are missing; that download is separate from model API usage.
