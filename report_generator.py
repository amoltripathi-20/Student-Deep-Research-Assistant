def generate_report(llm, context: str, query: str, sources=None) -> str:
    """Generate a structured Markdown research report."""
    sources = sources or []
    source_text = "\n".join(f"- {url}" for url in sources)
    if not source_text:
        source_text = "- Uploaded document / supplied context"

    prompt = f"""
You are an academic research assistant.
Create a clear, detailed, well-structured research report using only
the supplied context. Do not fabricate facts, statistics, or references.
If evidence is missing, mention the limitation.

Research topic: {query}

Context:
{context}

Use these sections:
# Introduction
# Key Concepts
# Detailed Explanation
# Applications
# Challenges and Limitations
# Future Scope
# Conclusion

Include a concise summary at the beginning.
Do not invent citations. The source list supplied separately by the app
will be shown below the report.

"""
    response = llm.invoke(prompt)
    report_body = getattr(response, "content", str(response))

    return (
        f"{report_body}\n\n"
        f"## Sources retrieved\n{source_text}\n"
    )
