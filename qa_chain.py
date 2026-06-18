from langchain_ollama import OllamaLLM


def load_llm():

    return OllamaLLM(
        model="llama3",
        base_url="http://host.docker.internal:11434"
    )


def ask_question(
    retriever,
    question
):

    docs = retriever.search(question)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
Answer ONLY using the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

    llm = load_llm()

    answer = llm.invoke(prompt)

    sources = []

    for doc in docs:

        page = doc.metadata.get(
            "page",
            0
        )

        source = doc.metadata.get(
            "source",
            "Unknown"
        )

        sources.append(
            {
                "page": page + 1,
                "source": source,
                "score": "Hybrid"
            }
        )

    return answer, sources