import pytest

from utils.pdf_processor import create_chunks, extract_text_from_pdf


def test_create_chunks_preserves_page_number_and_overlap():
    pages = [{"page_number": 3, "text": "one two three four five six seven eight"}]
    chunks = create_chunks(pages, chunk_size=4, overlap=2)

    assert [chunk["text"] for chunk in chunks] == [
        "one two three four",
        "three four five six",
        "five six seven eight",
    ]
    assert all(chunk["page_number"] == 3 for chunk in chunks)
    assert [chunk["chunk_id"] for chunk in chunks] == [0, 1, 2]


def test_chunks_do_not_cross_page_boundaries():
    pages = [
        {"page_number": 1, "text": "alpha beta gamma"},
        {"page_number": 4, "text": "delta epsilon"},
    ]
    chunks = create_chunks(pages, chunk_size=2, overlap=0)

    assert [chunk["page_number"] for chunk in chunks] == [1, 1, 4]
    assert chunks[-1]["text"] == "delta epsilon"


@pytest.mark.parametrize(
    "chunk_size,overlap",
    [(0, 0), (-1, 0), (5, -1), (5, 5), (5, 6)],
)
def test_invalid_chunk_settings_raise(chunk_size, overlap):
    with pytest.raises(ValueError):
        create_chunks([{"page_number": 1, "text": "some text"}], chunk_size, overlap)


def test_empty_pages_return_no_chunks():
    assert create_chunks([{"page_number": 1, "text": "   "}]) == []


def test_extract_text_from_pdf_accepts_bytes():
    import fitz

    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "RAG document test")
    pdf_bytes = document.tobytes()
    document.close()

    pages = extract_text_from_pdf(pdf_bytes)
    assert len(pages) == 1
    assert pages[0]["page_number"] == 1
    assert "RAG document test" in pages[0]["text"]
