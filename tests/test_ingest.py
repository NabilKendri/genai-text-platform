from langchain_core.documents import Document
from ingest import clean_text, split_documents


def test_clean_text_fixes_html_codes():
    assert clean_text("It#39;s   great ") == "It's great"


def test_clean_text_removes_extra_spaces():
    assert clean_text("  a \n\n b  ") == "a b"


def test_split_keeps_metadata():
    doc = Document(page_content="word " * 300, metadata={"label": "sports"})
    chunks = split_documents([doc], chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(c.metadata["label"] == "sports" for c in chunks)