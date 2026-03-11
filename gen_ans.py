from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from embedding_functions import llm

# ── Query rewriter ────────────────────────────────────────────────────────────
rewrite_system = """You are a search query optimizer.
Rewrite the user's question into a dense, keyword-rich search query that will 
retrieve the most relevant documents from a vector database.
Return ONLY the rewritten query, nothing else. No explanation."""

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", rewrite_system),
    ("human", "Original question: {question}\n\nRewritten search query:"),
])

rewrite_chain = rewrite_prompt | llm | StrOutputParser()


def rewrite_query(question: str) -> str:
    """Rewrite the user question into a better search query."""
    try:
        rewritten = rewrite_chain.invoke({"question": question}).strip()
        print(f"  [REWRITE] '{question}' → '{rewritten}'")
        return rewritten
    except Exception as e:
        print(f"  [REWRITE] Failed ({e}), using original")
        return question


# ── Answer generator ──────────────────────────────────────────────────────────
system = """You are an expert assistant for document question-answering.

Instructions:
- Answer ONLY from the provided documents. Do not use outside knowledge.
- If the documents do not contain the answer, say clearly: "The provided documents do not contain information about this topic."
- Use 3-6 sentences. Be thorough but concise.
- If multiple documents are relevant, synthesize them into one coherent answer.
- Do NOT include any citations, references, footnotes, or source markers in your answer (no [1], no [doc1], no [10†page 0], nothing like that).
- Write plain flowing text only. Citations are handled separately."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Retrieved documents:\n\n<docs>{documents}</docs>\n\nUser question: <question>{question}</question>"),
])

model     = llm
rag_chain = prompt | model | StrOutputParser()


def format_docs(docs):
    return "\n".join(
        f"<doc{i+1}>:\n"
        f"ID:{doc.metadata.get('id', 'N/A')}\n"
        f"Page:{doc.metadata.get('page', 'N/A')}\n"
        f"Source:{doc.metadata.get('source', 'N/A')}\n"
        f"Content:{doc.page_content}\n"
        f"</doc{i+1}>\n"
        for i, doc in enumerate(docs)
    )


def generate_res(docs_to_use, question):
    return rag_chain.invoke({
        "documents": format_docs(docs_to_use),
        "question":  question,
    })