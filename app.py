import streamlit as st
from dotenv import load_dotenv

from utils.pdf_loader import (
    load_pdf,
    split_documents
)

from utils.vector_store import (
    create_vector_store
)

from utils.qa_chain import (
    ask_question
)

load_dotenv()

st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Enterprise RAG Knowledge Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None 

if "pages" not in st.session_state:
    st.session_state.pages = 0

if "chunks" not in st.session_state:
    st.session_state.chunks = 0

with st.sidebar:

    st.header("📊 Statistics")

    st.metric(
        "Pages Loaded",
        st.session_state.pages
    )

    st.metric(
        "Chunks Created",
        st.session_state.chunks
    )

    st.metric(
        "Messages",
        len(st.session_state.messages)
    )

    st.divider()

    if st.button("🗑 Clear Chat"):

        st.session_state.messages = []

        st.rerun()

uploaded_files = st.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    all_docs = []

    with st.spinner("Processing PDFs..."):

        for uploaded_file in uploaded_files:

            with open(
                uploaded_file.name,
                "wb"
            ) as f:

                f.write(
                    uploaded_file.getbuffer()
                )

            docs = load_pdf(
                uploaded_file.name
            )

            all_docs.extend(
                docs
            )

        st.write(
            f"Total Documents Loaded: {len(all_docs)}"
        )

        chunks = split_documents(
            all_docs
        )

        st.write(
            f"Total Chunks Created: {len(chunks)}"
        )

        if len(chunks) == 0:

            st.error(
                "No chunks were created."
            )

            st.stop()

        st.write("DEBUG")
        st.write("Documents:", len(all_docs))
        st.write("Chunks:", len(chunks))

        vector_db = create_vector_store(
            chunks
        )

        st.session_state.vector_db = (
            vector_db
        )

        st.session_state.pages = (
            len(all_docs)
        )

        st.session_state.chunks = (
            len(chunks)
        )

        from utils.hybrid_retriever import (
            HybridRetriever
        )

        retriever = HybridRetriever(
            chunks,
            vector_db
        )

        st.session_state.retriever = (
            retriever
        )

    st.success(
        "PDFs Processed Successfully"
    )

for msg in st.session_state.messages:

    with st.chat_message(
        msg["role"]
    ):

        st.markdown(
            msg["content"]
        )

if (
    st.session_state.vector_db is not None
    and st.session_state.retriever is not None
):

    prompt = st.chat_input(
        "Ask a question..."
    )

    if prompt:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        with st.chat_message(
            "user"
        ):
            st.markdown(prompt)

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Thinking..."
            ):

                answer, sources = (
                    ask_question(
                        st.session_state.retriever,
                        prompt
                    )
                )

                st.markdown(answer)

                st.divider()

                st.markdown(
                    "### 📚 Sources"
                )

                for source in sources:

                    st.write(
                        f"Page: {source['page']} | Score: {source['score']}"
                    )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )