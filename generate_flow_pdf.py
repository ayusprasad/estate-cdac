"""
DocuRAG — Full Architecture & Flow Chart Generator
Generates a comprehensive multi-page PDF with:
  Page 1: High-Level System Architecture
  Page 2: Document Ingestion Pipeline (detailed flow)
  Page 3: Retrieval & RAG Generation Pipeline (detailed flow)
  Page 4: Database Models & Relationships
  Page 5: API Layer & Frontend Architecture
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


# ── Colour Palette ──────────────────────────────────────────────────
C = {
    "bg":           "#0f1117",
    "card_dark":    "#1a1d27",
    "card_mid":     "#232738",
    "accent_blue":  "#3b82f6",
    "accent_cyan":  "#06b6d4",
    "accent_green": "#10b981",
    "accent_purple":"#8b5cf6",
    "accent_orange":"#f59e0b",
    "accent_pink":  "#ec4899",
    "accent_red":   "#ef4444",
    "text_white":   "#f1f5f9",
    "text_muted":   "#94a3b8",
    "border":       "#334155",
    "arrow":        "#64748b",
}


def draw_rounded_box(ax, x, y, w, h, label, color, fontsize=9, text_color="white", alpha=0.92, sublabel=None, border_color=None):
    """Draw a styled rounded rectangle with label."""
    bc = border_color or color
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02",
        facecolor=color, edgecolor=bc,
        linewidth=1.5, alpha=alpha, zorder=3
    )
    ax.add_patch(box)
    if sublabel:
        ax.text(x + w / 2, y + h * 0.62, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=text_color, zorder=4)
        ax.text(x + w / 2, y + h * 0.32, sublabel, ha="center", va="center",
                fontsize=fontsize - 2, color=C["text_muted"], zorder=4, style="italic")
    else:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=text_color, zorder=4,
                wrap=True)
    return box


def draw_arrow(ax, x1, y1, x2, y2, color=None, style="-|>", lw=1.5, connectionstyle="arc3,rad=0"):
    """Draw an arrow between two points."""
    arrow_color = color or C["arrow"]
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, color=arrow_color,
        lw=lw, mutation_scale=15, zorder=2,
        connectionstyle=connectionstyle
    )
    ax.add_patch(arrow)
    return arrow


def setup_page(fig, title, subtitle=None):
    """Create a styled page with dark background."""
    ax = fig.add_subplot(111)
    ax.set_facecolor(C["bg"])
    fig.set_facecolor(C["bg"])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    # Title
    ax.text(5, 9.65, title, ha="center", va="center", fontsize=18,
            fontweight="bold", color=C["text_white"], zorder=5)
    if subtitle:
        ax.text(5, 9.35, subtitle, ha="center", va="center", fontsize=10,
                color=C["text_muted"], zorder=5, style="italic")
    # Title underline
    ax.plot([1.5, 8.5], [9.2, 9.2], color=C["accent_blue"], lw=2, alpha=0.6, zorder=5)
    return ax


# ════════════════════════════════════════════════════════════════════
# PAGE 1: HIGH-LEVEL SYSTEM ARCHITECTURE
# ════════════════════════════════════════════════════════════════════
def page1_high_level(pdf):
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 Landscape
    ax = setup_page(fig, "DocuRAG — High-Level System Architecture",
                    "Enterprise AI Document Intelligence & RAG System")

    # ── User / Client Layer ─────────────────────────────────────
    draw_rounded_box(ax, 3.5, 8.3, 3, 0.6, "User / Web Browser", C["accent_purple"], fontsize=11)

    # Arrow down to frontend
    draw_arrow(ax, 5, 8.3, 5, 7.8, C["accent_cyan"], lw=2)

    # ── Frontend ────────────────────────────────────────────────
    draw_rounded_box(ax, 2.5, 7.2, 5, 0.6, "Frontend  (HTML/CSS/JS SPA)", C["card_mid"],
                     fontsize=10, border_color=C["accent_cyan"],
                     sublabel="Upload • Search • Chat • SQL • Health Dashboard")

    draw_arrow(ax, 5, 7.2, 5, 6.7, C["accent_cyan"], lw=2)

    # ── FastAPI Layer ───────────────────────────────────────────
    draw_rounded_box(ax, 1.5, 6.05, 7, 0.6, "FastAPI REST API Layer  (Uvicorn)", C["accent_blue"],
                     fontsize=11, sublabel="/api/v1/documents  •  /api/v1/search  •  /api/v1/chat  •  /api/v1/sql  •  /api/v1/health")

    # Arrows from API to processing modules
    draw_arrow(ax, 2.8, 6.05, 1.8, 5.55, C["accent_green"], lw=1.8)
    draw_arrow(ax, 5, 6.05, 5, 5.55, C["accent_orange"], lw=1.8)
    draw_arrow(ax, 7.2, 6.05, 8.2, 5.55, C["accent_pink"], lw=1.8)

    # ── Core Processing Modules ─────────────────────────────────
    # Ingestion Pipeline
    draw_rounded_box(ax, 0.3, 4.6, 3, 0.9, "Document Processing\nPipeline", C["accent_green"],
                     fontsize=10, alpha=0.85,
                     sublabel="Classify → Extract → Chunk → Embed")

    # Retrieval & RAG
    draw_rounded_box(ax, 3.5, 4.6, 3, 0.9, "Retrieval & RAG\nEngine", C["accent_orange"],
                     fontsize=10, alpha=0.85,
                     sublabel="Hybrid Search → Rerank → LLM Generate")

    # SQL Agent
    draw_rounded_box(ax, 6.7, 4.6, 3, 0.9, "SQL Database\nAgent", C["accent_pink"],
                     fontsize=10, alpha=0.85,
                     sublabel="NL→SQL → Execute → Format")

    # ── Sub-components row ──────────────────────────────────────
    # Ingestion sub-components
    subs_ingest = [
        ("Document\nClassifier", 0.3),
        ("Data\nExtractor", 1.15),
        ("Semantic\nChunker", 2.0),
        ("Vector\nEmbedder", 2.85),
    ]
    for label, xp in subs_ingest:
        draw_rounded_box(ax, xp, 3.65, 0.75, 0.7, label, C["card_mid"],
                         fontsize=7, border_color=C["accent_green"])

    # Retrieval sub-components
    subs_retrieval = [
        ("Search\nRouter", 3.55),
        ("Query\nPlanner", 4.35),
        ("Hybrid\nEngine", 5.15),
        ("Reranker", 5.95),
    ]
    for label, xp in subs_retrieval:
        draw_rounded_box(ax, xp, 3.65, 0.75, 0.7, label, C["card_mid"],
                         fontsize=7, border_color=C["accent_orange"])

    # SQL sub-components
    subs_sql = [
        ("Schema\nInspector", 6.75),
        ("Query\nBuilder", 7.55),
        ("Safe\nExecutor", 8.35),
    ]
    for label, xp in subs_sql:
        draw_rounded_box(ax, xp, 3.65, 0.75, 0.7, label, C["card_mid"],
                         fontsize=7, border_color=C["accent_pink"])

    # Arrows down from processing row
    draw_arrow(ax, 1.8, 4.6, 1.8, 4.35, C["accent_green"], lw=1.2)
    draw_arrow(ax, 5, 4.6, 5, 4.35, C["accent_orange"], lw=1.2)
    draw_arrow(ax, 8.0, 4.6, 8.0, 4.35, C["accent_pink"], lw=1.2)

    # ── LLM Layer ───────────────────────────────────────────────
    draw_rounded_box(ax, 3, 2.6, 4, 0.7, "LLM Generator", C["accent_purple"],
                     fontsize=10, sublabel="llama.cpp  •  Qwen2.5-14B-Instruct  •  GGUF Q4_K_M")
    draw_arrow(ax, 5, 3.65, 5, 3.3, C["accent_purple"], lw=1.5)

    # Citation Verifier
    draw_rounded_box(ax, 7.3, 2.7, 2.2, 0.5, "Citation Verifier", C["card_mid"],
                     fontsize=8, border_color=C["accent_red"],
                     sublabel="Faithfulness 0.0–1.0")
    draw_arrow(ax, 7, 2.95, 7.3, 2.95, C["accent_red"], lw=1.2)

    # Evaluator
    draw_rounded_box(ax, 0.5, 2.7, 2.2, 0.5, "Evaluator", C["card_mid"],
                     fontsize=8, border_color=C["accent_cyan"],
                     sublabel="P@K • MRR • NDCG • Latency")
    draw_arrow(ax, 3, 2.95, 2.7, 2.95, C["accent_cyan"], lw=1.2)

    # ── Data Layer ──────────────────────────────────────────────
    draw_arrow(ax, 5, 2.6, 5, 2.1, C["accent_blue"], lw=2)

    # PostgreSQL + pgvector
    draw_rounded_box(ax, 1.5, 1.1, 3.2, 0.9, "PostgreSQL 15\n+ pgvector", C["accent_blue"],
                     fontsize=11, sublabel="Relational + Vector Store")

    # File Store
    draw_rounded_box(ax, 5.3, 1.1, 2, 0.9, "File Store", C["card_mid"],
                     fontsize=10, border_color=C["accent_green"],
                     sublabel="data/raw • processed • temp")

    # Redis
    draw_rounded_box(ax, 7.7, 1.1, 1.8, 0.9, "Redis 7+", C["card_mid"],
                     fontsize=10, border_color=C["accent_orange"],
                     sublabel="Queue & Cache")

    # ── DB tables inside PostgreSQL ─────────────────────────────
    tables = ["Documents", "Pages", "Chunks\n(+embeddings)", "Processing\nJobs"]
    for i, t in enumerate(tables):
        draw_rounded_box(ax, 1.6 + i * 0.75, 0.25, 0.7, 0.6, t, C["card_dark"],
                         fontsize=6.5, border_color=C["accent_blue"])

    draw_arrow(ax, 3.1, 1.1, 3.1, 0.85, C["accent_blue"], lw=1)

    # ── Embedding Model ─────────────────────────────────────────
    draw_rounded_box(ax, 0.3, 0.25, 1.2, 0.6, "BAAI/bge-\nsmall-en-v1.5", C["card_dark"],
                     fontsize=7, border_color=C["accent_cyan"],
                     sublabel="Embeddings")

    # Footer
    ax.text(5, -0.15, "DocuRAG v1.0 — CPU-Optimised, Open-Source Enterprise Document Intelligence",
            ha="center", fontsize=7, color=C["text_muted"], style="italic")

    pdf.savefig(fig)
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════
# PAGE 2: DOCUMENT INGESTION PIPELINE
# ════════════════════════════════════════════════════════════════════
def page2_ingestion(pdf):
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = setup_page(fig, "Document Ingestion Pipeline",
                    "Phase 1–4: Upload → Classify → Extract → Chunk → Embed → Store")

    # ── Step 1: File Upload ─────────────────────────────────────
    draw_rounded_box(ax, 3.8, 8.4, 2.4, 0.55, "FILE UPLOAD", C["accent_blue"], fontsize=11)
    ax.text(5, 8.05, "POST /api/v1/documents/upload", ha="center", fontsize=7, color=C["text_muted"])

    # Supported formats
    formats = ["PDF", "DOCX", "XLSX", "CSV", "Image", "TXT"]
    for i, fmt in enumerate(formats):
        draw_rounded_box(ax, 1.2 + i * 1.3, 7.3, 1.1, 0.45, fmt, C["card_mid"],
                         fontsize=8, border_color=C["accent_cyan"])
    draw_arrow(ax, 5, 8.4, 5, 7.95, C["accent_cyan"], lw=2)

    # ── Step 2: Save Raw File ───────────────────────────────────
    draw_arrow(ax, 5, 7.3, 5, 7.0, C["accent_cyan"], lw=2)
    draw_rounded_box(ax, 3.5, 6.35, 3, 0.55, "Save Raw File + DB Record", C["card_mid"],
                     fontsize=9, border_color=C["accent_green"],
                     sublabel="SHA256 hash • file_operations.py")
    ax.text(8, 6.6, "→ data/raw/", ha="center", fontsize=7, color=C["accent_green"])

    # ── Step 3: Document Classifier ─────────────────────────────
    draw_arrow(ax, 5, 6.35, 5, 6.05, C["accent_green"], lw=2)
    draw_rounded_box(ax, 3, 5.3, 4, 0.65, "DOCUMENT CLASSIFIER", C["accent_green"],
                     fontsize=11, sublabel="document_classifier.py")

    # Classification outcomes
    outcomes = [
        ("Digital PDF", 1.0, C["accent_blue"]),
        ("Scanned PDF", 3.2, C["accent_orange"]),
        ("Mixed PDF", 5.4, C["accent_purple"]),
        ("DOCX / XLSX\n/ CSV / TXT", 7.6, C["accent_cyan"]),
    ]
    for label, xp, clr in outcomes:
        draw_rounded_box(ax, xp, 4.4, 1.6, 0.55, label, C["card_mid"],
                         fontsize=8, border_color=clr)

    draw_arrow(ax, 2.5, 5.3, 1.8, 4.95, C["accent_blue"], lw=1.2)
    draw_arrow(ax, 4, 5.3, 4, 4.95, C["accent_orange"], lw=1.2)
    draw_arrow(ax, 5.5, 5.3, 6.2, 4.95, C["accent_purple"], lw=1.2)
    draw_arrow(ax, 7, 5.3, 8.4, 4.95, C["accent_cyan"], lw=1.2)

    # ── Step 4: Data Extraction ─────────────────────────────────
    draw_rounded_box(ax, 3, 3.3, 4, 0.8, "DATA EXTRACTOR", C["accent_orange"],
                     fontsize=11, sublabel="data_extractor.py")

    # Extraction methods
    methods = [
        ("pdfplumber\n(Text+Tables)", 0.5, C["accent_blue"]),
        ("Tesseract\nOCR", 2.4, C["accent_orange"]),
        ("EasyOCR\n(Fallback)", 4.2, C["accent_orange"]),
        ("python-docx", 6.0, C["accent_cyan"]),
        ("openpyxl\n/ csv", 7.8, C["accent_cyan"]),
    ]
    for label, xp, clr in methods:
        draw_rounded_box(ax, xp, 2.3, 1.5, 0.6, label, C["card_dark"],
                         fontsize=7, border_color=clr)
    draw_arrow(ax, 5, 3.3, 5, 2.9, C["accent_orange"], lw=1.5)

    # Merge arrows
    for label, xp, clr in outcomes:
        draw_arrow(ax, xp + 0.8, 4.4, 5, 4.1, clr, lw=1)

    # ── Step 5: Semantic Chunker ────────────────────────────────
    draw_arrow(ax, 5, 2.3, 5, 2.0, C["accent_purple"], lw=2)
    draw_rounded_box(ax, 3, 1.3, 4, 0.6, "SEMANTIC CHUNKER", C["accent_purple"],
                     fontsize=11, sublabel="semantic_chunker.py  •  Paragraph/Sentence boundaries  •  Overlap")

    # ── Step 6: Vector Embedder ─────────────────────────────────
    draw_arrow(ax, 5, 1.3, 5, 1.0, C["accent_cyan"], lw=2)
    draw_rounded_box(ax, 2.5, 0.2, 5, 0.7, "VECTOR EMBEDDER  ->  PostgreSQL + pgvector",
                     C["accent_cyan"], fontsize=10,
                     sublabel="BAAI/bge-small-en-v1.5  •  384-dim embeddings  •  Stored in Chunk.embedding")

    # Output arrow to DB
    draw_rounded_box(ax, 8, 0.3, 1.5, 0.5, "STORED", C["accent_green"],
                     fontsize=9)
    draw_arrow(ax, 7.5, 0.55, 8, 0.55, C["accent_green"], lw=2)

    pdf.savefig(fig)
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════
# PAGE 3: RETRIEVAL & RAG PIPELINE
# ════════════════════════════════════════════════════════════════════
def page3_retrieval_rag(pdf):
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = setup_page(fig, "Retrieval & RAG Generation Pipeline",
                    "Phase 5–9: Query → Route → Retrieve → Rerank → Generate → Cite")

    # ── User Query ──────────────────────────────────────────────
    draw_rounded_box(ax, 3.5, 8.4, 3, 0.55, "USER QUERY", C["accent_blue"], fontsize=11)
    ax.text(5, 8.05, "POST /api/v1/chat  or  POST /api/v1/search", ha="center", fontsize=7, color=C["text_muted"])

    # ── Search Router (Intent Classifier) ───────────────────────
    draw_arrow(ax, 5, 8.4, 5, 8.0, C["accent_cyan"], lw=2)
    draw_rounded_box(ax, 3, 7.2, 4, 0.65, "SEARCH ROUTER (Intent Classifier)", C["accent_purple"],
                     fontsize=10, sublabel="search_router.py  •  9 Intent Categories")

    # Intent categories
    intents = ["FACTUAL", "ANALYTICAL", "NUMERICAL", "TABULAR", "SQL_DATA",
               "MULTI-\nLINGUAL", "SUMMARI-\nSATION", "IMAGE", "GENERAL"]
    for i, intent in enumerate(intents):
        x = 0.3 + i * 1.05
        draw_rounded_box(ax, x, 6.35, 0.95, 0.5, intent, C["card_dark"],
                         fontsize=6, border_color=C["accent_purple"])

    draw_arrow(ax, 5, 7.2, 5, 6.95, C["accent_purple"], lw=1.5)

    # ── Decision Fork ───────────────────────────────────────────
    # SQL path (right)
    draw_arrow(ax, 7, 7.2, 8.5, 6.0, C["accent_pink"], lw=2)
    ax.text(8.2, 6.8, "SQL_DATA\nIntent", ha="center", fontsize=7, color=C["accent_pink"], fontweight="bold")

    draw_rounded_box(ax, 7.5, 5.0, 2, 0.85, "SQL AGENT", C["accent_pink"],
                     fontsize=10, sublabel="sql_agent.py\nNL→SQL→Execute")

    # SQL sub-steps
    sql_steps = ["Schema\nInspect", "Build\nSQL", "Validate\n& Execute", "Format\nResults"]
    for i, s in enumerate(sql_steps):
        draw_rounded_box(ax, 7.6 + i * 0.45, 4.1, 0.4, 0.6, s, C["card_dark"],
                         fontsize=5.5, border_color=C["accent_pink"])
    draw_arrow(ax, 8.5, 5.0, 8.5, 4.7, C["accent_pink"], lw=1)

    # Document retrieval path (left)
    draw_arrow(ax, 3, 7.2, 2, 6.0, C["accent_orange"], lw=2)
    ax.text(1.8, 6.8, "Other\nIntents", ha="center", fontsize=7, color=C["accent_orange"], fontweight="bold")

    # ── Query Planner ───────────────────────────────────────────
    draw_rounded_box(ax, 0.5, 5.2, 3.5, 0.6, "QUERY PLANNER", C["accent_orange"],
                     fontsize=10, sublabel="query_planner.py  •  Multi-step decomposition")

    # ── Hybrid Retrieval Engine ─────────────────────────────────
    draw_arrow(ax, 2.25, 5.2, 2.25, 4.85, C["accent_orange"], lw=2)
    draw_rounded_box(ax, 0.5, 3.8, 3.5, 0.9, "HYBRID RETRIEVAL ENGINE", C["accent_green"],
                     fontsize=10, sublabel="engine.py  •  Reciprocal Rank Fusion")

    # Two retrieval paths
    draw_rounded_box(ax, 0.6, 2.9, 1.5, 0.6, "Vector Search\n(Cosine Sim)", C["card_mid"],
                     fontsize=7, border_color=C["accent_cyan"])
    draw_rounded_box(ax, 2.4, 2.9, 1.5, 0.6, "BM25 Keyword\nSearch", C["card_mid"],
                     fontsize=7, border_color=C["accent_orange"])
    draw_arrow(ax, 1.35, 3.8, 1.35, 3.5, C["accent_cyan"], lw=1.2)
    draw_arrow(ax, 3.15, 3.8, 3.15, 3.5, C["accent_orange"], lw=1.2)

    # ── Reranker ────────────────────────────────────────────────
    draw_arrow(ax, 2.25, 2.9, 2.25, 2.55, C["accent_purple"], lw=2)
    draw_rounded_box(ax, 0.5, 1.8, 3.5, 0.65, "CROSS-ENCODER RERANKER", C["accent_purple"],
                     fontsize=10, sublabel="ms-marco-MiniLM-L-6-v2")

    # ── Merge paths ─────────────────────────────────────────────
    # Both paths converge to LLM Generator
    draw_arrow(ax, 2.25, 1.8, 5, 1.45, C["accent_purple"], lw=2)
    draw_arrow(ax, 8.5, 4.1, 5, 1.45, C["accent_pink"], lw=2)

    # ── LLM Generator ──────────────────────────────────────────
    draw_rounded_box(ax, 3.5, 0.6, 3, 0.75, "LLM GENERATOR", C["accent_blue"],
                     fontsize=11, sublabel="llama.cpp  •  Qwen2.5-14B  •  Citation-aware prompts")

    # ── Citation Verifier ───────────────────────────────────────
    draw_arrow(ax, 6.5, 0.97, 7.3, 0.97, C["accent_red"], lw=2)
    draw_rounded_box(ax, 7.3, 0.6, 2.2, 0.75, "CITATION\nVERIFIER", C["accent_red"],
                     fontsize=9, sublabel="Faithfulness Score")

    # ── Final Output ────────────────────────────────────────────
    draw_arrow(ax, 5, 0.6, 5, 0.3, C["accent_green"], lw=2)
    ax.text(5, 0.1, "Answer + Citations + Faithfulness Score -> User",
            ha="center", fontsize=9, color=C["accent_green"], fontweight="bold")

    pdf.savefig(fig)
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════
# PAGE 4: DATABASE MODELS & RELATIONSHIPS
# ════════════════════════════════════════════════════════════════════
def page4_database(pdf):
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = setup_page(fig, "Database Models & Data Relationships",
                    "SQLAlchemy 2.0 (Async) + PostgreSQL 15 + pgvector")

    # ── Document Model ──────────────────────────────────────────
    draw_rounded_box(ax, 0.5, 5.5, 3.5, 3.2, "", C["card_mid"],
                     border_color=C["accent_blue"])
    ax.text(2.25, 8.3, "Document", ha="center", fontsize=13,
            fontweight="bold", color=C["accent_blue"])
    fields_doc = [
        "id  (UUID PK)",
        "filename  (String)",
        "file_path  (String)",
        "file_hash  (SHA256)",
        "doc_type  (Enum: PDF/DOCX/...)",
        "status  (Enum: PENDING/...)",
        "total_pages  (Integer)",
        "metadata  (JSON)",
        "created_at  (DateTime)",
        "updated_at  (DateTime)",
    ]
    for i, f in enumerate(fields_doc):
        ax.text(0.7, 7.95 - i * 0.24, f"• {f}", fontsize=7,
                color=C["text_white"], family="monospace")

    # ── Page Model ──────────────────────────────────────────────
    draw_rounded_box(ax, 4.5, 5.5, 3, 3.2, "", C["card_mid"],
                     border_color=C["accent_green"])
    ax.text(6, 8.3, "Page", ha="center", fontsize=13,
            fontweight="bold", color=C["accent_green"])
    fields_page = [
        "id  (UUID PK)",
        "document_id  (FK → Document)",
        "page_number  (Integer)",
        "raw_text  (Text)",
        "tables_json  (JSON)",
        "layout_json  (JSON)",
        "language  (String)",
        "is_scanned  (Boolean)",
        "created_at  (DateTime)",
    ]
    for i, f in enumerate(fields_page):
        ax.text(4.7, 7.95 - i * 0.24, f"• {f}", fontsize=7,
                color=C["text_white"], family="monospace")

    # ── Chunk Model ─────────────────────────────────────────────
    draw_rounded_box(ax, 4.5, 1.8, 3, 3.2, "", C["card_mid"],
                     border_color=C["accent_purple"])
    ax.text(6, 4.6, "Chunk", ha="center", fontsize=13,
            fontweight="bold", color=C["accent_purple"])
    fields_chunk = [
        "id  (UUID PK)",
        "document_id  (FK → Document)",
        "page_id  (FK → Page)",
        "chunk_text  (Text)",
        "chunk_index  (Integer)",
        "embedding  (Vector 384-dim)",
        "token_count  (Integer)",
        "metadata  (JSON)",
        "created_at  (DateTime)",
    ]
    for i, f in enumerate(fields_chunk):
        ax.text(4.7, 4.25 - i * 0.24, f"• {f}", fontsize=7,
                color=C["text_white"], family="monospace")

    # ── ProcessingJob Model ─────────────────────────────────────
    draw_rounded_box(ax, 0.5, 1.8, 3.5, 3.2, "", C["card_mid"],
                     border_color=C["accent_orange"])
    ax.text(2.25, 4.6, "ProcessingJob", ha="center", fontsize=13,
            fontweight="bold", color=C["accent_orange"])
    fields_job = [
        "id  (UUID PK)",
        "document_id  (FK → Document)",
        "status  (Enum: PENDING/...)",
        "progress  (Float 0.0–1.0)",
        "current_step  (String)",
        "error_message  (Text)",
        "started_at  (DateTime)",
        "completed_at  (DateTime)",
        "processing_stats  (JSON)",
    ]
    for i, f in enumerate(fields_job):
        ax.text(0.7, 4.25 - i * 0.24, f"• {f}", fontsize=7,
                color=C["text_white"], family="monospace")

    # ── Relationships ───────────────────────────────────────────
    # Document → Page (1:N)
    draw_arrow(ax, 4.0, 7.0, 4.5, 7.0, C["accent_cyan"], lw=2.5)
    ax.text(4.25, 7.2, "1 : N", ha="center", fontsize=8, color=C["accent_cyan"], fontweight="bold")

    # Document → Chunk (1:N)
    draw_arrow(ax, 2.25, 5.5, 4.5, 4.0, C["accent_cyan"], lw=2.5,
               connectionstyle="arc3,rad=-0.3")
    ax.text(3.0, 4.5, "1 : N", ha="center", fontsize=8, color=C["accent_cyan"], fontweight="bold")

    # Page → Chunk (1:N)
    draw_arrow(ax, 6.0, 5.5, 6.0, 5.0, C["accent_cyan"], lw=2.5)
    ax.text(6.3, 5.25, "1 : N", ha="center", fontsize=8, color=C["accent_cyan"], fontweight="bold")

    # Document → ProcessingJob (1:N)
    draw_arrow(ax, 2.25, 5.5, 2.25, 5.0, C["accent_cyan"], lw=2.5)
    ax.text(2.7, 5.25, "1 : N", ha="center", fontsize=8, color=C["accent_cyan"], fontweight="bold")

    # ── Enums box ───────────────────────────────────────────────
    draw_rounded_box(ax, 7.8, 5.5, 1.8, 3.2, "", C["card_dark"],
                     border_color=C["border"])
    ax.text(8.7, 8.3, "Enums", ha="center", fontsize=11,
            fontweight="bold", color=C["text_muted"])
    enums = [
        "DocumentStatus:",
        "  PENDING",
        "  PROCESSING",
        "  COMPLETED",
        "  FAILED",
        "",
        "DocumentType:",
        "  PDF",
        "  DOCX",
        "  XLSX",
        "  CSV",
        "  IMAGE",
        "  TXT",
    ]
    for i, e in enumerate(enums):
        ax.text(7.95, 7.95 - i * 0.18, e, fontsize=6.5,
                color=C["text_muted"], family="monospace")

    # pgvector note
    draw_rounded_box(ax, 7.8, 2.0, 1.8, 2.8, "", C["card_dark"],
                     border_color=C["accent_cyan"])
    ax.text(8.7, 4.45, "pgvector", ha="center", fontsize=11,
            fontweight="bold", color=C["accent_cyan"])
    pgv_info = [
        "Extension for",
        "PostgreSQL that",
        "enables vector",
        "similarity search.",
        "",
        "Chunk.embedding",
        "stores 384-dim",
        "float vectors.",
        "",
        "Cosine similarity",
        "used for nearest",
        "neighbour search.",
    ]
    for i, line in enumerate(pgv_info):
        ax.text(7.95, 4.1 - i * 0.18, line, fontsize=6.5,
                color=C["text_muted"])

    # Footer
    ax.text(5, 1.2, "All models defined in  src/database_models/  •  Migrations via Alembic  •  Async sessions via SQLAlchemy 2.0",
            ha="center", fontsize=7, color=C["text_muted"], style="italic")

    pdf.savefig(fig)
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════
# PAGE 5: API LAYER & FRONTEND
# ════════════════════════════════════════════════════════════════════
def page5_api_frontend(pdf):
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = setup_page(fig, "API Layer & Frontend Architecture",
                    "FastAPI REST Endpoints + Single-Page Application")

    # ── API Endpoints ───────────────────────────────────────────
    draw_rounded_box(ax, 0.3, 4.5, 9.4, 4.5, "", C["card_dark"],
                     border_color=C["accent_blue"])
    ax.text(5, 8.75, "FastAPI REST API  —  /api/v1/", ha="center", fontsize=14,
            fontweight="bold", color=C["accent_blue"])

    # Health routes
    draw_rounded_box(ax, 0.5, 7.8, 2.6, 0.7, "", C["card_mid"], border_color=C["accent_green"])
    ax.text(1.8, 8.25, "/health", ha="center", fontsize=10, fontweight="bold", color=C["accent_green"])
    health_endpoints = ["GET /health  → System status"]
    for i, ep in enumerate(health_endpoints):
        ax.text(0.7, 7.98 - i * 0.18, ep, fontsize=6.5, color=C["text_white"], family="monospace")

    # Document routes
    draw_rounded_box(ax, 3.3, 7.2, 3.0, 1.3, "", C["card_mid"], border_color=C["accent_orange"])
    ax.text(4.8, 8.25, "/documents", ha="center", fontsize=10, fontweight="bold", color=C["accent_orange"])
    doc_endpoints = [
        "POST   /upload    → Ingest file",
        "GET    /          → List all docs",
        "GET    /{id}      → Doc details",
        "DELETE /{id}      → Remove doc",
    ]
    for i, ep in enumerate(doc_endpoints):
        ax.text(3.5, 7.98 - i * 0.18, ep, fontsize=6.5, color=C["text_white"], family="monospace")

    # Search routes
    draw_rounded_box(ax, 6.5, 7.8, 3.0, 0.7, "", C["card_mid"], border_color=C["accent_cyan"])
    ax.text(8, 8.25, "/search", ha="center", fontsize=10, fontweight="bold", color=C["accent_cyan"])
    search_endpoints = [
        "POST /          → Hybrid search",
        "GET  /stats      → Index stats",
    ]
    for i, ep in enumerate(search_endpoints):
        ax.text(6.7, 7.98 - i * 0.18, ep, fontsize=6.5, color=C["text_white"], family="monospace")

    # Chat routes
    draw_rounded_box(ax, 0.5, 6.3, 2.6, 0.7, "", C["card_mid"], border_color=C["accent_purple"])
    ax.text(1.8, 6.75, "/chat", ha="center", fontsize=10, fontweight="bold", color=C["accent_purple"])
    chat_endpoints = [
        "POST /  → RAG-powered chat",
    ]
    for i, ep in enumerate(chat_endpoints):
        ax.text(0.7, 6.48 - i * 0.18, ep, fontsize=6.5, color=C["text_white"], family="monospace")

    # SQL routes
    draw_rounded_box(ax, 3.3, 5.1, 3.0, 1.9, "", C["card_mid"], border_color=C["accent_pink"])
    ax.text(4.8, 6.75, "/sql", ha="center", fontsize=10, fontweight="bold", color=C["accent_pink"])
    sql_endpoints = [
        "POST   /connect        → Register DB",
        "DELETE /connect/{label} → Deregister",
        "GET    /connections     → List DBs",
        "GET    /schema/{label}  → Inspect",
        "POST   /query           → NL→SQL",
    ]
    for i, ep in enumerate(sql_endpoints):
        ax.text(3.5, 6.48 - i * 0.18, ep, fontsize=6.5, color=C["text_white"], family="monospace")

    # Eval routes
    draw_rounded_box(ax, 6.5, 6.3, 3.0, 0.7, "", C["card_mid"], border_color=C["accent_red"])
    ax.text(8, 6.75, "/eval", ha="center", fontsize=10, fontweight="bold", color=C["accent_red"])
    eval_endpoints = [
        "POST /run  → Quality benchmarks",
        "GET  /results  → Eval metrics",
    ]
    for i, ep in enumerate(eval_endpoints):
        ax.text(6.7, 6.48 - i * 0.18, ep, fontsize=6.5, color=C["text_white"], family="monospace")

    # ── Request/Response flow ───────────────────────────────────
    ax.text(5, 4.85, "JSON Request/Response  •  CORS enabled  •  Swagger /docs  •  ReDoc /redoc",
            ha="center", fontsize=7, color=C["text_muted"], style="italic")

    # ── Frontend ────────────────────────────────────────────────
    draw_rounded_box(ax, 0.3, 0.3, 9.4, 3.7, "", C["card_dark"],
                     border_color=C["accent_cyan"])
    ax.text(5, 3.75, "Frontend — Single Page Application", ha="center", fontsize=14,
            fontweight="bold", color=C["accent_cyan"])

    # Frontend tabs
    tabs = [
        ("Upload", "Drag & drop\nfile upload\nprogress bar", C["accent_orange"]),
        ("Search", "Hybrid search\nresults with\ncitations", C["accent_green"]),
        ("Chat", "Conversational\nRAG with\nfaithfulness", C["accent_purple"]),
        ("SQL", "Connect DBs\nNL query\nschema inspect", C["accent_pink"]),
        ("Health", "System metrics\nCPU/Memory\nmodel status", C["accent_red"]),
    ]
    for i, (title, desc, clr) in enumerate(tabs):
        x = 0.6 + i * 1.85
        draw_rounded_box(ax, x, 1.5, 1.6, 1.8, "", C["card_mid"], border_color=clr)
        ax.text(x + 0.8, 3.0, title, ha="center", fontsize=9, fontweight="bold", color=clr)
        ax.text(x + 0.8, 2.3, desc, ha="center", fontsize=7, color=C["text_white"])

    # Tech stack
    ax.text(5, 1.0, "index.html  +  css/index.css  +  js/app.js",
            ha="center", fontsize=8, color=C["text_muted"], family="monospace")
    ax.text(5, 0.65, "Dark Theme  •  Glassmorphism  •  Responsive  •  Served as Static Files by FastAPI",
            ha="center", fontsize=7, color=C["text_muted"], style="italic")

    # Connection arrow
    draw_arrow(ax, 5, 4.5, 5, 4.1, C["accent_cyan"], lw=2.5)
    ax.text(5.5, 4.3, "HTTP / JSON", ha="center", fontsize=7, color=C["accent_cyan"])

    pdf.savefig(fig)
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════
# PAGE 6: COMPLETE FILE TREE MAP
# ════════════════════════════════════════════════════════════════════
def page6_file_tree(pdf):
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = setup_page(fig, "Project File Tree & Module Map",
                    "Complete directory structure with module responsibilities")

    tree_lines = [
        ("estate/  (DocuRAG Root)", 0, C["accent_blue"], True),
        ("├── application_configuration/", 1, C["accent_orange"], True),
        ("│   ├── environment_settings.py     Pydantic Settings (.env)", 2, C["text_muted"], False),
        ("│   └── logger_setup.py             Structlog / logging config", 2, C["text_muted"], False),
        ("├── src/", 1, C["accent_green"], True),
        ("│   ├── main_application.py         FastAPI app entrypoint", 2, C["accent_green"], False),
        ("│   ├── api/", 2, C["accent_blue"], True),
        ("│   │   ├── v1_router.py            API v1 router aggregator", 3, C["text_muted"], False),
        ("│   │   └── routes/", 3, C["accent_blue"], True),
        ("│   │       ├── health_routes.py    GET /health", 4, C["text_muted"], False),
        ("│   │       ├── document_routes.py  CRUD + upload endpoints", 4, C["text_muted"], False),
        ("│   │       ├── search_routes.py    Hybrid search endpoints", 4, C["text_muted"], False),
        ("│   │       ├── chat_routes.py      RAG chat endpoint", 4, C["text_muted"], False),
        ("│   │       └── sql_routes.py       SQL agent endpoints", 4, C["text_muted"], False),
        ("│   ├── database_models/", 2, C["accent_purple"], True),
        ("│   │   ├── database_connection.py  Async engine + sessions", 3, C["text_muted"], False),
        ("│   │   ├── shared_enums.py         Status & Type enums", 3, C["text_muted"], False),
        ("│   │   ├── document_model.py       Document ORM model", 3, C["text_muted"], False),
        ("│   │   ├── page_model.py           Page ORM model", 3, C["text_muted"], False),
        ("│   │   ├── chunk_model.py          Chunk + embedding ORM", 3, C["text_muted"], False),
        ("│   │   └── processing_job_model.py Job tracking ORM", 3, C["text_muted"], False),
        ("│   ├── document_processing/", 2, C["accent_orange"], True),
        ("│   │   ├── ingestion_pipeline.py   Orchestrator", 3, C["text_muted"], False),
        ("│   │   ├── document_classifier.py  Digital/Scanned/Mixed", 3, C["text_muted"], False),
        ("│   │   ├── data_extractor.py       Multi-format extraction", 3, C["text_muted"], False),
        ("│   │   ├── semantic_chunker.py     Boundary-aware chunking", 3, C["text_muted"], False),
        ("│   │   ├── vector_embedder.py      BGE embedding + store", 3, C["text_muted"], False),
        ("│   │   └── processing_schemas.py   Pydantic DTOs", 3, C["text_muted"], False),
        ("│   ├── retrieval/", 2, C["accent_cyan"], True),
        ("│   │   ├── engine.py               Hybrid retrieval + RRF", 3, C["text_muted"], False),
        ("│   │   ├── retriever.py            Vector + BM25 retrievers", 3, C["text_muted"], False),
        ("│   │   ├── reranker.py             Cross-encoder reranker", 3, C["text_muted"], False),
        ("│   │   ├── search_router.py        9-intent classifier", 3, C["text_muted"], False),
        ("│   │   ├── query_planner.py        Multi-step decomposition", 3, C["text_muted"], False),
        ("│   │   ├── search_service.py       High-level orchestrator", 3, C["text_muted"], False),
        ("│   │   ├── sql_agent.py            NL→SQL agent", 3, C["text_muted"], False),
        ("│   │   ├── citation_verifier.py    Faithfulness scoring", 3, C["text_muted"], False),
        ("│   │   └── evaluator.py            P@K, MRR, NDCG metrics", 3, C["text_muted"], False),
        ("│   ├── llm/", 2, C["accent_pink"], True),
        ("│   │   └── generator.py            llama.cpp + extractive", 3, C["text_muted"], False),
        ("│   └── shared_utilities/", 2, C["text_muted"], True),
        ("│       ├── file_operations.py      File I/O + hashing", 3, C["text_muted"], False),
        ("│       └── text_cleaning.py        Text normalization", 3, C["text_muted"], False),
        ("├── frontend/", 1, C["accent_cyan"], True),
        ("│   ├── index.html                  SPA shell", 2, C["text_muted"], False),
        ("│   ├── css/index.css               Dark glassmorphism UI", 2, C["text_muted"], False),
        ("│   └── js/app.js                   Frontend logic", 2, C["text_muted"], False),
        ("├── alembic/                        DB migrations", 1, C["text_muted"], False),
        ("├── tests/                          Pytest suite", 1, C["text_muted"], False),
        ("├── scripts/                        Setup scripts", 1, C["text_muted"], False),
        ("├── data/  (raw/ processed/ temp/)  Runtime data", 1, C["text_muted"], False),
        ("├── models/                         GGUF LLM weights", 1, C["text_muted"], False),
        ("└── logs/                           Runtime logs", 1, C["text_muted"], False),
    ]

    y_start = 8.95
    line_h = 0.165
    for i, (text, indent, color, is_bold) in enumerate(tree_lines):
        y = y_start - i * line_h
        if y < 0.2:
            break
        x = 0.5 + indent * 0.4
        weight = "bold" if is_bold else "normal"
        ax.text(x, y, text, fontsize=6.5, color=color, family="monospace",
                fontweight=weight)

    pdf.savefig(fig)
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════
# MAIN — Generate the PDF
# ════════════════════════════════════════════════════════════════════
def main():
    output_path = r"c:\Users\kumar\Desktop\estate\flow_new.pdf"
    print(f"Generating DocuRAG architecture PDF -> {output_path}")

    with PdfPages(output_path) as pdf:
        page1_high_level(pdf)
        print("  [OK] Page 1: High-Level Architecture")
        page2_ingestion(pdf)
        print("  [OK] Page 2: Ingestion Pipeline")
        page3_retrieval_rag(pdf)
        print("  [OK] Page 3: Retrieval & RAG Pipeline")
        page4_database(pdf)
        print("  [OK] Page 4: Database Models")
        page5_api_frontend(pdf)
        print("  [OK] Page 5: API & Frontend")
        page6_file_tree(pdf)
        print("  [OK] Page 6: File Tree Map")

    print(f"\nDone! PDF saved to: {output_path}")


if __name__ == "__main__":
    main()
