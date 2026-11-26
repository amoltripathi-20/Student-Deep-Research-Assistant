import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from utils import web_search, scrape_website, read_pdf
from rag_pipeline import create_vector_store, retrieve_context
from report_generator import generate_report

load_dotenv()

st.set_page_config(
    page_title="Student Deep Research Assistant",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Student Deep Research Assistant")
st.markdown("AI-powered research using **RAG + web sources + uploaded PDFs**")


def get_secret(name: str, default: str = "") -> str:
    """Read a secret from Streamlit Cloud first, then environment variables."""
    try:
        value = st.secrets.get(name, default)
        if value:
            return str(value).strip()
    except Exception:
        # st.secrets may not exist in a local environment without secrets.toml.
        pass
    return os.getenv(name, default).strip()


groq_api_key = get_secret("GROQ_API_KEY")
groq_model = get_secret("GROQ_MODEL", "llama-3.3-70b-versatile")

if not groq_api_key:
    st.error(
        "GROQ_API_KEY is missing. Add it in Streamlit Cloud → "
        "Manage app → Settings → Secrets, or create a local .env file."
    )
    st.stop()

try:
    llm = ChatGroq(
        api_key=groq_api_key,
        model=groq_model,
        temperature=0.3,
    )
except Exception as exc:
    st.error(f"Could not initialize the Groq client: {exc}")
    st.stop()

query = st.text_input("Enter your research topic")
pdf_file = st.file_uploader("Upload a PDF for additional knowledge", type=["pdf"])
run_eval = st.checkbox("Enable performance evaluation")

if st.button("Start Research", type="primary"):
    if not query.strip() and pdf_file is None:
        st.warning("Enter a research topic or upload a PDF.")
        st.stop()

    progress = st.progress(0)
    status = st.empty()
    collected_text = []
    source_urls = []

    # Web search is optional when the user only wants to query an uploaded PDF.
    if query.strip():
        status.write("🔎 Searching the web...")
        try:
            source_urls = web_search(query, max_results=5)
        except Exception as exc:
            st.warning(f"Web search is temporarily unavailable: {exc}")
            source_urls = []
    progress.progress(20)

    status.write("🌐 Extracting website content...")
    for url in source_urls:
        try:
            page_text = scrape_website(url)
            if page_text:
                collected_text.append(f"Source URL: {url}\n{page_text}")
        except Exception as exc:
            st.warning(f"Could not read {url}: {exc}")
    progress.progress(40)

    if pdf_file is not None:
        status.write("📄 Extracting PDF text...")
        try:
            pdf_text = read_pdf(pdf_file)
            if pdf_text.strip():
                collected_text.append(
                    f"Uploaded PDF: {pdf_file.name}\n{pdf_text}"
                )
            else:
                st.warning("No selectable text was found in the PDF. "
                           "Scanned PDFs may require OCR.")
        except Exception as exc:
            st.error(f"Could not read the uploaded PDF: {exc}")
            st.stop()
    progress.progress(60)

    full_text = "\n\n".join(collected_text).strip()
    if not full_text:
        st.error(
            "No content was retrieved. Try another search query or upload "
            "a text-based PDF. If web search is blocked, try again later."
        )
        st.stop()

    try:
        status.write("🧠 Building the knowledge base...")
        vector_db = create_vector_store(full_text)
        progress.progress(80)

        search_query = query.strip() or "Summarize the uploaded document"
        context = retrieve_context(vector_db, search_query)

        if not context.strip():
            st.error("The retriever did not return any relevant content.")
            st.stop()

        status.write("🤖 Generating the research answer...")
        answer_prompt = f"""
You are a careful academic research assistant.
Answer the user's question using the supplied context.
Do not invent facts that are not supported by the context.
If the context is insufficient, clearly say what is missing.
Use clear headings and explain important concepts.

Context:
{context}

User question:
{search_query}
"""
        answer = llm.invoke(answer_prompt).content
        report = generate_report(llm, context, search_query, source_urls)

        progress.progress(100)
        status.success("Research completed.")

        st.subheader("📚 Research Answer")
        st.markdown(answer)

        st.subheader("📑 Structured Research Report")
        st.markdown(report)

        if run_eval:
            try:
                from evaluation import evaluate_rag
                with st.spinner("Evaluating the generated answer..."):
                    metrics = evaluate_rag(
                        search_query,
                        answer,
                        [context],
                    )
                st.subheader("📊 Performance Metrics")
                st.json(metrics)
            except Exception as exc:
                st.warning(f"Evaluation could not be completed: {exc}")

        st.subheader("🔗 Sources")
        if source_urls:
            for url in source_urls:
                st.markdown(f"- [{url}]({url})")
        elif pdf_file is not None:
            st.write(f"Uploaded document: {pdf_file.name}")
        else:
            st.write("No external sources were retrieved.")

        st.download_button(
            label="Download Research Report",
            data=report,
            file_name="research_report.md",
            mime="text/markdown",
        )

    except Exception as exc:
        st.error(f"Research pipeline failed: {exc}")
        st.exception(exc)
