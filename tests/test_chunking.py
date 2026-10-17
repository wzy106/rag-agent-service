from langchain_core.documents import Document

from rag_core.chunking import (
    derive_chunk_id,
    enrich_chunks,
    section_path_for_offset,
    split_sections,
    stable_chunk_id,
)


def test_stable_chunk_id_is_deterministic():
    a = stable_chunk_id("a.md", "Intro > API", "hello", 0)
    b = stable_chunk_id("a.md", "Intro > API", "hello", 0)
    assert a == b
    assert len(a) == 16


def test_same_content_across_sources_gets_distinct_ids():
    a = stable_chunk_id("a.md", "", "same text", 0)
    b = stable_chunk_id("b.md", "", "same text", 0)
    assert a != b


def test_same_content_same_source_occurrence_disambiguates():
    a = stable_chunk_id("a.md", "", "same text", 0)
    b = stable_chunk_id("a.md", "", "same text", 1)
    assert a != b


def test_section_path_ignores_headings_in_code_fence():
    text = "# Root\n\n```md\n# not a heading\n```\n\n## Real\n\nbody\n"
    paths = [path for _, path in split_sections(text)]
    assert "Root" in paths
    assert "Root > Real" in paths
    assert all("not a heading" not in path for path in paths)
    assert section_path_for_offset(text, text.index("body")) == "Root > Real"


def test_enrich_chunks_sets_ids_and_section_paths():
    source = "# Title\n\nintro\n\n## Part\n\nbody text\n"
    chunks = [
        Document(page_content="intro", metadata={"source": "a.md", "start_index": 8}),
        Document(page_content="body text", metadata={"source": "a.md", "start_index": 27}),
    ]
    out = enrich_chunks(chunks, {"a.md": source})
    assert [c.metadata["chunk_index"] for c in out] == [0, 1]
    assert out[0].metadata["section_path"] == "Title"
    assert out[1].metadata["section_path"] == "Title > Part"
    assert len({c.metadata["chunk_id"] for c in out}) == 2


def test_enrich_chunks_same_content_occurrence():
    source = "same\n\nsame\n"
    chunks = [
        Document(page_content="same", metadata={"source": "a.md"}),
        Document(page_content="same", metadata={"source": "a.md"}),
    ]
    out = enrich_chunks(chunks, {"a.md": source})
    assert out[0].metadata["chunk_id"] != out[1].metadata["chunk_id"]


def test_derive_chunk_id_for_legacy_doc():
    doc = Document(page_content="legacy", metadata={"source": "old.md"})
    assert len(derive_chunk_id(doc)) == 16
    doc.metadata["chunk_id"] = "fixed"
    assert derive_chunk_id(doc) == "fixed"
