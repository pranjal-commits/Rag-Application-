from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf(pdf_path):

    loader = PyPDFLoader(
        pdf_path
    )

    docs = loader.load()

    for doc in docs:

        doc.metadata[
            "source"
        ] = pdf_path

    return docs

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    return chunks


docs = load_pdf("Pranjal_office_executive.pdf")
chunks = split_documents(docs)

print(f"Pages Loaded: {len(docs)}")
print(f"Chunks Created: {len(chunks)}")

