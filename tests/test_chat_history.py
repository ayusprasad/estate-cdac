"""
Unit test for chat history logging feature
"""
import json
from pathlib import Path
from src.shared_utilities.chat_history_logger import log_chat_interaction, CHAT_HISTORY_JSONL, CHAT_HISTORY_MD

def test_log_chat_interaction(tmp_path, monkeypatch):
    test_jsonl = tmp_path / "chat_history.jsonl"
    test_md = tmp_path / "chat_history.md"
    
    monkeypatch.setattr("src.shared_utilities.chat_history_logger.CHAT_HISTORY_JSONL", test_jsonl)
    monkeypatch.setattr("src.shared_utilities.chat_history_logger.CHAT_HISTORY_MD", test_md)
    
    log_chat_interaction(
        query="What is AI?",
        answer="AI stands for Artificial Intelligence.",
        mode="llm",
        intent="factual",
        faithfulness=0.95,
        citations=[{"rank": 1, "document_name": "ai_paper.pdf", "page_number": 2}],
        latency_ms=120.5
    )
    
    assert test_jsonl.exists()
    assert test_md.exists()
    
    with open(test_jsonl, "r", encoding="utf-8") as f:
        data = json.loads(f.readline())
        assert data["question"] == "What is AI?"
        assert data["answer"] == "AI stands for Artificial Intelligence."
        assert data["mode"] == "llm"
        assert data["faithfulness"] == 0.95
        assert data["citations"][0]["document_name"] == "ai_paper.pdf"
        
    with open(test_md, "r", encoding="utf-8") as f:
        content = f.read()
        assert "What is AI?" in content
        assert "Artificial Intelligence" in content
        assert "ai_paper.pdf" in content
