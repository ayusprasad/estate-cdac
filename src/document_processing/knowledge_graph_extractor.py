"""
DocuRAG — Phase 11: Knowledge Graph Extractor

Builds a NetworkX Knowledge Graph from document chunks using SpaCy NLP.
Extracts Entities (ORG, PERSON, GPE, DATE) and connects them to the document chunks,
allowing for GraphRAG reasoning (e.g., finding documents sharing the same entities).
"""
from __future__ import annotations

import networkx as nx
from pathlib import Path
import spacy
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database_models.chunk_model import Chunk
from src.database_models.document_model import Document
from application_configuration.logger_setup import get_logger
from application_configuration.environment_settings import get_settings

logger = get_logger(__name__)
settings = get_settings()

# Load spacy model globally to avoid reloading
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

async def build_knowledge_graph(document_id: str, db: AsyncSession) -> None:
    """
    Extracts entities from all chunks in a document and updates the global Knowledge Graph.
    Saves the graph as a GraphML file in the data/processed/ directory.
    """
    logger.info("Starting Knowledge Graph extraction", document_id=str(document_id))
    
    # 1. Fetch document and chunks
    doc_result = await db.execute(select(Document).where(Document.id == document_id))
    document = doc_result.scalar_one_or_none()
    
    if not document:
        logger.error("Document not found for KG extraction", document_id=str(document_id))
        return
        
    chunk_result = await db.execute(select(Chunk).where(Chunk.document_id == document_id))
    chunks = chunk_result.scalars().all()
    
    # 2. Load or initialize the global NetworkX Graph
    graph_path = Path(settings.storage.processed_dir) / "global_knowledge_graph.graphml"
    
    if graph_path.exists():
        try:
            G = nx.read_graphml(graph_path)
        except Exception as e:
            logger.warning("Failed to read existing graphml, creating new", exc_info=e)
            G = nx.Graph()
    else:
        G = nx.Graph()
        
    # 3. Add Document Node
    doc_node_id = f"DOC_{document.id}"
    G.add_node(doc_node_id, type="Document", title=document.original_filename)
    
    # 4. Extract Entities per chunk
    entity_count = 0
    for chunk in chunks:
        chunk_node_id = f"CHUNK_{chunk.id}"
        G.add_node(chunk_node_id, type="Chunk", text=chunk.text[:50] + "...")
        G.add_edge(doc_node_id, chunk_node_id, relation="CONTAINS")
        
        doc_spacy = nlp(chunk.text)
        for ent in doc_spacy.ents:
            # Filter for relevant entity types to avoid noise
            if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT", "EVENT"]:
                ent_id = f"ENT_{ent.text.upper()}"
                G.add_node(ent_id, type="Entity", entity_type=ent.label_, name=ent.text)
                # Link chunk to entity
                G.add_edge(chunk_node_id, ent_id, relation="MENTIONS")
                entity_count += 1
                
    # 5. Save the updated Graph
    nx.write_graphml(G, graph_path)
    logger.info("Knowledge Graph extraction complete", 
                document_id=str(document_id), 
                entities_extracted=entity_count,
                total_nodes=G.number_of_nodes())
