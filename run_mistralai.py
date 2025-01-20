import os
import glob
from typing import List
from langchain_community.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain


api_key = os.getenv("mistralai_API_KEY")

def load_single_document(file_path: str) -> List[Document]:
    loader = PyPDFLoader(file_path)
    docs = loader.load()    
    return docs


all_files = glob.glob(os.path.join("source_documents", f"**/*.pdf"), recursive=True)
documents = []
for docs in all_files:
    documents.extend(load_single_document(docs))

text_splitter = CharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=400
)
texts = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings()
vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever()
model = ChatMistralAI(mistral_api_key=api_key)
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(model, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

while True:
    query = input("\nEnter a query: ")
    if query == "exit":
        break
    if query.strip() == "":
        continue
    response = retrieval_chain.invoke({"input": query})
    print(response["answer"])