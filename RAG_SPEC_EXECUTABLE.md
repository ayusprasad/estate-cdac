# DocuRAG Executable Spec v2.0

**DO THIS**: 250 PDFs → zero false negatives + exact citations + metrics.json proof.  
**Constraint**: Offline CPU (i5-12500H, 16GB). No OpenAI/Claude APIs.  
**Standard**: Answer miss docs = system says "not in corpus" (no hallucinate).

---

## TASK 1: PDF Parser + Metadata (pdfplumber + Tesseract OCR fallback)

**What**: Extract text, tables, page layout, section hierarchy per PDF. Assign doc_id, page nums, section paths.

**DO THIS**:
```bash
pip install pdfplumber pypdf pytesseract pillow rank_bm25 --break-system-packages
# Install Tesseract OS package (fallback for scanned PDFs)
```

**Code (python/pdf_parser.py)**:
```python
import pdfplumber, json, hashlib
from pathlib import Path

def ingest_pdf(pdf_path, doc_id=None):
    """Extract: text, tables, metadata. Assign doc_id, page num, section."""
    if not doc_id:
        doc_id = hashlib.md5(Path(pdf_path).read_bytes()).hexdigest()[:8]
    
    doc = {"doc_id": doc_id, "filename": Path(pdf_path).name, "pages": []}
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            page_text = page.extract_text() or ""
            tables = page.extract_tables() or []
            
            # Fallback OCR if text extraction sparse
            if len(page_text.strip()) < 100 and tables == []:
                page_text = ocr_page(page)  # See below
            
            doc["pages"].append({
                "page_num": page_num,
                "text": page_text,
                "tables": tables,
                "layout_type": detect_layout_type(page)
            })
    
    return doc

def ocr_page(page):
    """Tesseract OCR fallback for scanned PDFs."""
    import pytesseract
    try:
        pil_img = page.to_image().original
        return pytesseract.image_to_string(pil_img)
    except:
        return ""

def detect_layout_type(page):
    """Guess page type: text, table, mixed."""
    text = page.extract_text() or ""
    tables = page.extract_tables() or []
    if len(tables) > 2: return "table"
    if len(text) > 500: return "text"
    return "mixed" if text and tables else "image"

# Usage
doc = ingest_pdf("sample.pdf", doc_id="doc_001")
print(json.dumps(doc, indent=2))  # Verify structure
```

**TEST**:
```bash
# Parse 10 PDFs. Verify: all pages extracted, tables intact, no missing text.
# Save to /mnt/agents/data/ingested_pdfs/
```

---

## TASK 2: Hierarchical Chunker (Section → Paragraph → Sentence)

**What**: Break PDFs into 3 levels. Level 2 = 256-512 token chunks for embedding.

**DO THIS**:
```python
# File: chunker.py
import re
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")

def chunk_document(doc, chunk_size=384, overlap_pct=0.12):
    """
    Level 1: Section (coarse routing)
    Level 2: Semantic paragraph (embedding)
    Level 3: Sentence/table-row (citation anchor)
    """
    chunks = []
    chunk_id = 0
    
    for page in doc["pages"]:
        page_num = page["page_num"]
        text = page["text"]
        tables = page["tables"]
        
        # Split by sentence
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Group into Level-2 chunks (target: chunk_size tokens)
        current_chunk_tokens = []
        current_chunk_text = ""
        
        for sentence in sentences:
            sent_tokens = tokenizer.encode(sentence)
            
            if len(current_chunk_tokens) + len(sent_tokens) > chunk_size:
                # Save chunk
                if current_chunk_text:
                    chunks.append({
                        "chunk_id": f"{doc['doc_id']}_p{page_num}_c{chunk_id}",
                        "doc_id": doc["doc_id"],
                        "page_num": page_num,
                        "text": current_chunk_text.strip(),
                        "token_count": len(current_chunk_tokens),
                        "chunk_type": "text"
                    })
                    chunk_id += 1
                current_chunk_tokens = []
                current_chunk_text = ""
            
            current_chunk_tokens.extend(sent_tokens)
            current_chunk_text += " " + sentence
        
        # Final chunk
        if current_chunk_text:
            chunks.append({
                "chunk_id": f"{doc['doc_id']}_p{page_num}_c{chunk_id}",
                "doc_id": doc["doc_id"],
                "page_num": page_num,
                "text": current_chunk_text.strip(),
                "token_count": len(current_chunk_tokens),
                "chunk_type": "text"
            })
            chunk_id += 1
        
        # Table chunks (embed whole table if <300 tokens)
        for table_idx, table in enumerate(tables):
            table_text = "\n".join([" | ".join(map(str, row)) for row in table])
            table_tokens = tokenizer.encode(table_text)
            
            if len(table_tokens) < 300:
                chunks.append({
                    "chunk_id": f"{doc['doc_id']}_p{page_num}_t{table_idx}",
                    "doc_id": doc["doc_id"],
                    "page_num": page_num,
                    "text": table_text,
                    "token_count": len(table_tokens),
                    "chunk_type": "table"
                })
            else:
                # Split table by row
                for row_idx, row in enumerate(table):
                    row_text = " | ".join(map(str, row))
                    chunks.append({
                        "chunk_id": f"{doc['doc_id']}_p{page_num}_t{table_idx}_r{row_idx}",
                        "doc_id": doc["doc_id"],
                        "page_num": page_num,
                        "text": row_text,
                        "token_count": len(tokenizer.encode(row_text)),
                        "chunk_type": "table_row"
                    })
    
    return chunks

# Usage
chunks = chunk_document(doc)
print(f"Total chunks: {len(chunks)}, avg tokens: {sum(c['token_count'] for c in chunks) / len(chunks):.0f}")
```

**TEST**:
```
Expected: ~400 chunks per 100-page PDF.
Verify: No chunk > 512 tokens. No sentence split mid-word.
```

---

## TASK 3: Embedding + pgvector Index

**What**: Embed chunks. Store in PostgreSQL with HNSW index.

**DO THIS**:
```bash
# Install deps
pip install sentence-transformers psycopg2-binary --break-system-packages

# Start PostgreSQL (if not running)
sudo service postgresql start

# Create DB
psql postgres -c "CREATE DATABASE ragdb;"
psql ragdb -c "CREATE EXTENSION IF NOT EXISTS vector;"
psql ragdb -c "
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    page_num INT NOT NULL,
    chunk_type TEXT,
    text TEXT NOT NULL,
    embedding vector(384),
    token_count INT
);
CREATE INDEX IF NOT EXISTS idx_embedding ON chunks USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=64);
CREATE INDEX IF NOT EXISTS idx_doc_id ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_page ON chunks(page_num);
"
```

**Code (embedding.py)**:
```python
from sentence_transformers import SentenceTransformer
import psycopg2
import numpy as np

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def embed_and_index_chunks(chunks):
    """Embed chunk text. Store in pgvector."""
    conn = psycopg2.connect("dbname=ragdb user=postgres")
    cur = conn.cursor()
    
    for chunk in chunks:
        embedding = model.encode(chunk["text"], convert_to_numpy=True)
        embedding_list = embedding.tolist()
        
        cur.execute("""
            INSERT INTO chunks (chunk_id, doc_id, page_num, chunk_type, text, embedding, token_count)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (chunk_id) DO NOTHING;
        """, (
            chunk["chunk_id"],
            chunk["doc_id"],
            chunk["page_num"],
            chunk["chunk_type"],
            chunk["text"],
            embedding_list,
            chunk["token_count"]
        ))
    
    conn.commit()
    cur.close()
    conn.close()
    print(f"Indexed {len(chunks)} chunks.")

# Usage
embed_and_index_chunks(chunks)
```

**TEST**:
```bash
psql ragdb -c "SELECT COUNT(*) FROM chunks;"  # Should match chunk count
psql ragdb -c "SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL;"  # All embedded?
```

---

## TASK 4: Hybrid Search (Dense + BM25)

**What**: Combine vector similarity + keyword search. Retrieve top 50, rerank to top 10.

**Code (search.py)**:
```python
from rank_bm25 import BM25Okapi
import psycopg2
import numpy as np

class HybridSearch:
    def __init__(self, db_conn_str):
        self.conn_str = db_conn_str
        self.bm25 = None
        self._load_bm25_corpus()
    
    def _load_bm25_corpus(self):
        """Load all chunk texts for BM25."""
        conn = psycopg2.connect(self.conn_str)
        cur = conn.cursor()
        cur.execute("SELECT chunk_id, text FROM chunks;")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        texts = [row[1].split() for row in rows]
        self.bm25 = BM25Okapi(texts)
    
    def search(self, query, top_k=50, rerank_k=10):
        """
        1. Vector search (dense)
        2. BM25 search (sparse)
        3. Fuse scores
        4. Rerank with cross-encoder
        """
        conn = psycopg2.connect(self.conn_str)
        cur = conn.cursor()
        
        # Dense search
        query_embed = model.encode(query, convert_to_numpy=True).tolist()
        cur.execute("""
            SELECT chunk_id, text, 1 - (embedding <=> %s) AS similarity
            FROM chunks
            ORDER BY similarity DESC
            LIMIT %s;
        """, (query_embed, top_k))
        dense_results = {row[0]: {"text": row[1], "score": row[2]} for row in cur.fetchall()}
        
        # BM25 search
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        cur.execute("SELECT chunk_id FROM chunks ORDER BY chunk_id;")
        chunk_ids = [row[0] for row in cur.fetchall()]
        bm25_results = {chunk_ids[i]: {"score": bm25_scores[i]} for i in range(len(chunk_ids))}
        
        # Fuse (0.7 dense + 0.3 BM25)
        fused = {}
        for cid in set(dense_results.keys()) | set(bm25_results.keys()):
            d_score = dense_results.get(cid, {}).get("score", 0)
            b_score = bm25_results.get(cid, {}).get("score", 0)
            fused[cid] = 0.7 * d_score + 0.3 * (b_score / max(bm25_scores) if max(bm25_scores) > 0 else 0)
        
        # Sort + rerank top_k with cross-encoder
        top_chunks = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:rerank_k]
        cur.execute("SELECT text FROM chunks WHERE chunk_id = ANY(%s);", ([c[0] for c in top_chunks],))
        
        cur.close()
        conn.close()
        
        return top_chunks  # (chunk_id, fused_score)

# Usage
searcher = HybridSearch("dbname=ragdb user=postgres")
results = searcher.search("What is company X acquisition strategy?", rerank_k=10)
for chunk_id, score in results:
    print(f"{chunk_id}: {score:.3f}")
```

**TEST**:
```
Query: factual question from a test PDF.
Verify: Answer chunk in top 10. Score > 0.5.
```

---

## TASK 5: Local LLM + Citations

**What**: Use Qwen2.5-14B GGUF. Generate answer + cite exact page/section.

**DO THIS**:
```bash
# Download Qwen2.5-14B Q4_K_M
pip install llama-cpp-python --break-system-packages
# wget https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GGUF/...

# Or use ollama
curl https://ollama.ai/install.sh | sh
ollama pull qwen2.5:14b-instruct-q4_K_M
```

**Code (generator.py)**:
```python
from llama_cpp import Llama

llm = Llama(
    model_path="./Qwen2.5-14B-Instruct-Q4_K_M.gguf",
    n_ctx=2048,
    n_gpu_layers=0,  # CPU only
    n_threads=8,
    verbose=False
)

def generate_answer(query, retrieved_chunks):
    """Generate answer from chunks. Cite page/section."""
    
    context = "\n\n".join([
        f"[Source: {c['doc_id']}, Page {c['page_num']}]\n{c['text']}"
        for c in retrieved_chunks[:5]  # Top 5 chunks
    ])
    
    prompt = f"""You are a document assistant. Answer using ONLY the provided documents.
    
QUERY: {query}

DOCUMENTS:
{context}

ANSWER (cite as [Source: DOC_ID, Page X]):
If you cannot find the answer in the documents, say exactly: "Information not found in corpus."
"""
    
    response = llm(prompt, max_tokens=500, temperature=0.1)
    return response["choices"][0]["text"].strip()

# Usage
retrieved = [
    {"doc_id": "doc_001", "page_num": 5, "text": "Company X was acquired by Y in 2020."},
    {"doc_id": "doc_001", "page_num": 6, "text": "Acquisition price: $50 million."}
]
answer = generate_answer("When was company X acquired?", retrieved)
print(answer)
# Expected: "Company X was acquired by Y in 2020. [Source: doc_001, Page 5]"
```

**TEST**:
```
Question: Known answer in top 5 chunks.
Verify: Answer appears. Citation format correct. No hallucination.
```

---

## TASK 6: Citation Verification

**What**: Check if answer facts match cited sources. Threshold: 0.6 lexical overlap.

**Code (citation_verifier.py)**:
```python
from rouge_score import rouge_scorer

def verify_answer(answer, retrieved_chunks):
    """Score answer faithfulness. Return score + grounding map."""
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    # Split answer into sentences
    answer_sents = answer.split(". ")
    grounding = {}
    
    for sent in answer_sents:
        best_match_score = 0
        best_chunk_id = None
        
        for chunk in retrieved_chunks:
            score = scorer.score(sent, chunk["text"])["rougeL"].fmeasure
            if score > best_match_score:
                best_match_score = score
                best_chunk_id = chunk["chunk_id"]
        
        grounding[sent[:50] + "..."] = {
            "chunk_id": best_chunk_id,
            "overlap_score": best_match_score
        }
    
    avg_score = sum(g["overlap_score"] for g in grounding.values()) / len(grounding) if grounding else 0
    
    return {
        "avg_faithfulness": avg_score,
        "grounding": grounding,
        "is_verified": avg_score >= 0.6
    }

# Usage
result = verify_answer(answer, retrieved)
print(f"Faithfulness: {result['avg_faithfulness']:.2f}")
if not result["is_verified"]:
    print("WARNING: Answer not well-grounded. Trigger second retrieval pass.")
```

**TEST**:
```
Known true answer + false answer.
Verify: True answer scores > 0.6. False answer < 0.6.
```

---

## TASK 7: Eval Pipeline + Metrics

**What**: Run 100+ labeled QA pairs. Measure recall, accuracy, citation quality.

**Code (eval.py)**:
```python
import json
from datetime import datetime

def evaluate_rag(qa_pairs, searcher, generator, verifier):
    """
    qa_pairs = [{"query": "...", "doc_id": "...", "page": 5, "answer": "..."}, ...]
    """
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "corpus_stats": {
            "total_documents": 250,
            "total_pages": 24700,
            "total_chunks": 98500
        },
        "retrieval": {
            "recall_at_10": 0,
            "precision_at_10": 0,
            "mrr": 0,
            "false_negative_rate": 0
        },
        "answer_quality": {
            "accuracy": 0,
            "faithfulness": 0,
            "hallucination_rate": 0
        },
        "citation": {
            "citation_accuracy": 0,
            "citation_granularity": 0
        },
        "performance": {
            "p50_latency_ms": 0,
            "p95_latency_ms": 0
        }
    }
    
    correct_retrieval = 0
    correct_answer = 0
    correct_citation = 0
    hallucinations = 0
    latencies = []
    
    for qa in qa_pairs:
        query = qa["query"]
        ground_doc_id = qa["doc_id"]
        ground_page = qa["page"]
        ground_answer = qa["answer"]
        
        # Retrieve
        import time
        t0 = time.time()
        retrieved = searcher.search(query, rerank_k=10)
        latency = (time.time() - t0) * 1000
        latencies.append(latency)
        
        # Check if correct doc in top 10
        retrieved_doc_ids = [get_doc_id(c[0]) for c in retrieved]
        if ground_doc_id in retrieved_doc_ids:
            correct_retrieval += 1
            mrr_rank = retrieved_doc_ids.index(ground_doc_id) + 1
            results["retrieval"]["mrr"] += 1.0 / mrr_rank
        
        # Generate
        retrieved_texts = [fetch_chunk_text(c[0]) for c in retrieved[:5]]
        answer = generator.generate_answer(query, retrieved_texts)
        
        # Verify
        verification = verifier.verify_answer(answer, retrieved_texts)
        if verification["is_verified"]:
            correct_answer += 1
        
        # Check citation accuracy
        if f"[Source: {ground_doc_id}" in answer or f"Page {ground_page}" in answer:
            correct_citation += 1
        
        # Detect hallucination (answer contains factual claim not in retrieved)
        if "according to" in answer.lower() and not any(kw in retrieved_texts[0] for kw in ["according", "found"]):
            hallucinations += 1
    
    n = len(qa_pairs)
    results["retrieval"]["recall_at_10"] = correct_retrieval / n
    results["retrieval"]["false_negative_rate"] = 1 - (correct_retrieval / n)
    results["answer_quality"]["accuracy"] = correct_answer / n
    results["answer_quality"]["hallucination_rate"] = hallucinations / n
    results["citation"]["citation_accuracy"] = correct_citation / n
    results["performance"]["p50_latency_ms"] = sorted(latencies)[n//2]
    results["performance"]["p95_latency_ms"] = sorted(latencies)[int(n*0.95)]
    
    results["scores"] = compute_overall_score(results)
    
    return results

def compute_overall_score(results):
    """Weighted score. A+ = 90+."""
    r_quality = results["retrieval"]["recall_at_10"] * 100
    a_quality = results["answer_quality"]["accuracy"] * 100
    c_quality = results["citation"]["citation_accuracy"] * 100
    
    overall = 0.35 * r_quality + 0.30 * a_quality + 0.35 * c_quality
    
    return {
        "retrieval_quality": r_quality,
        "answer_quality": a_quality,
        "citation_quality": c_quality,
        "overall_rag_score": overall,
        "grade": "A+" if overall >= 90 else "A" if overall >= 80 else "B" if overall >= 70 else "C" if overall >= 60 else "D"
    }

# Usage
qa_test_set = [
    {"query": "When was company X acquired?", "doc_id": "doc_001", "page": 5, "answer": "2020"},
    # ... 100+ more
]
results = evaluate_rag(qa_test_set, searcher, generator, verifier)

# Save metrics
with open("/mnt/agents/output/metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Overall Score: {results['scores']['overall_rag_score']:.1f} ({results['scores']['grade']})")
```

**TEST**:
```bash
python eval.py
cat /mnt/agents/output/metrics.json
# Expected: overall_rag_score >= 80 (grade A or higher)
```

---

## TASK 8: Full Pipeline Integration

**What**: Wire all components. Run end-to-end on 250 PDFs.

**Code (main.py)**:
```python
from pdf_parser import ingest_pdf
from chunker import chunk_document
from embedding import embed_and_index_chunks
from search import HybridSearch
from generator import generate_answer
from citation_verifier import verify_answer
from eval import evaluate_rag

import glob
from pathlib import Path

# 1. Ingest all PDFs
pdf_paths = glob.glob("/mnt/agents/data/pdfs/*.pdf")
print(f"Found {len(pdf_paths)} PDFs.")

all_chunks = []
for pdf_path in pdf_paths:
    doc = ingest_pdf(pdf_path)
    chunks = chunk_document(doc)
    all_chunks.extend(chunks)
    print(f"  {Path(pdf_path).name}: {len(chunks)} chunks")

print(f"Total chunks: {len(all_chunks)}")

# 2. Embed + index
embed_and_index_chunks(all_chunks)

# 3. Search + generate on test set
searcher = HybridSearch("dbname=ragdb user=postgres")
generator = GeneratorQwen()
verifier = CitationVerifier()

qa_pairs = load_test_set("/mnt/agents/data/qa_test_set.jsonl")  # 100+ hand-labeled QA pairs
results = evaluate_rag(qa_pairs, searcher, generator, verifier)

# 4. Save metrics
import json
with open("/mnt/agents/output/metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n=== FINAL SCORE: {results['scores']['overall_rag_score']:.1f} ({results['scores']['grade']}) ===")
print(json.dumps(results["scores"], indent=2))
```

**RUN**:
```bash
python main.py
cat /mnt/agents/output/metrics.json
```

---

## CHECKLIST

- [ ] PDF parser: Extract text + tables + metadata. Test on 10 PDFs.
- [ ] Chunker: 256-512 token chunks. No sentence split. Test chunk count ~4x per 100 pages.
- [ ] Embedding: bghe-small-en-v1.5 + pgvector HNSW index. Test query embedding.
- [ ] Hybrid search: Dense + BM25 fusion. Test on known answer in corpus.
- [ ] LLM: Qwen2.5-14B Q4_K_M. Generate + cite. Test citation format.
- [ ] Citation verifier: ROUGE-L overlap. Threshold 0.6. Test on true/false answers.
- [ ] Eval pipeline: 100+ QA pairs. Measure recall, accuracy, citation quality.
- [ ] Metrics.json: Auto-save after eval. Grade A+ = overall score >= 90.
- [ ] **FINAL TEST**: Run main.py on 250 PDFs. Achieve grade A (overall >= 80).

---

## CONSTRAINTS (non-negotiable)

1. **Offline** — No OpenAI, Claude, or cloud embed APIs.
2. **CPU-first** — All models GGUF. Fit on i5-12500H + 16GB.
3. **No hallucination** — Answer missing docs → "Information not found in corpus."
4. **Exact citations** — Every claim cite [Doc ID, Page X].
5. **Measured** — Metrics.json proof. No untested changes.

---

*DocuRAG v2.0 — Built for AI/Claude Code execution.*
