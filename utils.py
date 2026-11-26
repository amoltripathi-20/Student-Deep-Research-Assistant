import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from pypdf import PdfReader
from urllib.parse import urlparse


def web_search(query: str, max_results: int = 5) -> list[str]:
    """
    Search the web with DDGS and return unique result URLs.

    DDGS returns a list of dictionaries (including the `href` field);
    do not parse the old LangChain tool's formatted string output.
    """
    query = (query or "").strip()
    if not query:
        return []

    try:
        results = DDGS().text(query, max_results=max_results)
        urls = []
        seen = set()

        for result in results or []:
            url = result.get("href") or result.get("url")
            if not url:
                continue

            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                continue

            if url not in seen:
                seen.add(url)
                urls.append(url)

            if len(urls) >= max_results:
                break

        return urls

    except Exception as exc:
        # Let the Streamlit UI display a helpful message and continue
        # with a PDF-only workflow if a PDF was uploaded.
        raise RuntimeError(
            "DuckDuckGo search failed or returned no results. "
            "This can be temporary rate-limiting/blocking on cloud hosting."
        ) from exc


def scrape_website(url: str) -> str:
    """Download a page and return its readable text."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("Only valid HTTP/HTTPS URLs can be scraped.")

    response = requests.get(
        url,
        timeout=(5, 20),
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
            )
        },
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(
        ["script", "style", "nav", "footer", "header", "aside", "noscript"]
    ):
        tag.decompose()

    return soup.get_text(separator=" ", strip=True)


def read_pdf(uploaded_file) -> str:
    """Extract selectable text from a Streamlit-uploaded PDF."""
    uploaded_file.seek(0)
    reader = PdfReader(uploaded_file)

    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(f"[Page {page_number}]\n{page_text}")

    return "\n\n".join(pages)
