import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from pypdf import PdfReader

DATA_PATH = "./papers"
DB_PATH = "faiss_index"


def load_pdfs(folder):
    docs = []
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(folder, file))
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if not text.strip():
                    continue

                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": file, "page": i + 1}
                    )
                )
    return docs


def main():
    docs = load_pdfs(DATA_PATH)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = []
    for doc in docs:
        split_docs = splitter.split_documents([doc])
        chunks.extend(split_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    db = FAISS.from_documents(chunks, embeddings)

    db.save_local(DB_PATH)
    print(f"Indexed {len(chunks)} chunks successfully!")


if __name__ == "__main__":
    main()