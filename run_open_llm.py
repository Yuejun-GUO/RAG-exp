import os
import glob
from typing import List
from langchain_community.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM


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

mdoel_id = "meta-llama/Llama-3.2-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(mdoel_id)
tokenizer = AutoTokenizer.from_pretrained(mdoel_id)
tokenizer.pad_token_id = tokenizer.eos_token_id

READER_LLM = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    do_sample=True,
    temperature=0.2,
    repetition_penalty=1.1,
    return_full_text=False,
    max_new_tokens=500,
)
prompt_in_chat_format = [
    {
        "role": "system",
        "content": """Using the information contained in the context, give a comprehensive answer to the question.
        Respond only to the question asked, response should be concise and relevant to the question.
        Provide the number of the source document when relevant.
        If the answer cannot be deduced from the context, do not give an answer.""",
    },
    {
        "role": "user",
        "content": """Context:
        {context}
        ---
        Now here is the question you need to answer.

        Question: {question}""",
    },
]
RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(
    prompt_in_chat_format, tokenize=False, add_generation_prompt=True
)

while True:
    query = input("\nEnter a query: ")
    if query == "exit":
        break
    if query.strip() == "":
        continue

    final_prompt = RAG_PROMPT_TEMPLATE.format(question=query, context=None)
    answer = READER_LLM(final_prompt)[0]["generated_text"]  
    print(answer)