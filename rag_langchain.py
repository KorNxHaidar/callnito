import sys
import re
import pandas as pd
import os
import time
import shutil
from pathlib import Path
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

FILE_PATH = Path(r"./thairath_new\thairath_articles_with_content.csv")
CHROMA_PATH = "./chroma_db_thairath"

def clean_text(text):
    text = str(text)
    text = re.sub(r'[\n\r\t]+', ' ', text)
    text = re.sub(r'\"\"\"', '\"', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

print("\n--- 1. Loading and Cleaning Data ---")
try:
    df = pd.read_csv(FILE_PATH)
    df.dropna(subset=['title', 'content'], inplace=True)
    df.drop_duplicates(subset=['content'], inplace=True)
    
    df['cleaned_title'] = df['title'].apply(clean_text)
    df['cleaned_content'] = df['content'].apply(clean_text)
    
    # Create a combined field for better context
    df['document_for_rag'] = "Title: " + df['cleaned_title'] + "\nContent: " + df['cleaned_content']
    print(f"Data loaded. Total rows: {len(df)}")
except Exception as e:
    print(f"Error loading file: {e}")
    sys.exit()

# --- 2. Chunking ---
print("\n--- 2. Splitting Text (Chunking) ---")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    add_start_index=True,
)

all_chunks = []
for index, row in df.iterrows():
    doc_text = row['document_for_rag']
    link = row['link']
    
    # Attempt to separate title and content for metadata
    try:
        parts = doc_text.split('\nContent: ', 1)
        title = parts[0].replace("Title: ", "")
        content = parts[1]
    except:
        title = "Unknown"
        content = doc_text

    chunks = text_splitter.split_text(content)
    
    for i, chunk_content in enumerate(chunks):
        # Prepend title to every chunk for context
        final_chunk_text = f"Title: {title}\nContent: {chunk_content}"
        
        chunk_metadata = {
            "source": link,
            "title": title,
            "chunk_id": f"doc_{index}_chunk_{i}"
        }
        all_chunks.append(Document(page_content=final_chunk_text, metadata=chunk_metadata))

print(f"Total chunks created: {len(all_chunks)}")

# --- 4. Show Example Results (แสดงตัวอย่างผลลัพธ์) ---
print("\n--- 4. ตัวอย่าง Chunks ---")
for i, chunk in enumerate(all_chunks[:6]):
    print(f"\n--- Chunk {i+1} ---")
    print(f"Metadata: {chunk.metadata}")
    print("\nPage Content:")
    print(f"'{chunk.page_content[:1000]}'") 
    print("-" * 20)


print("\n--- Loading Embedding Model (BAAI/bge-m3) ---")
# 2.1 ตั้งค่า Embedding Model
model_name = "BAAI/bge-m3"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}

try:
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    print(" Loaded bge-m3 model Successful. ")
except Exception as e:
    print(f"  [Error] เกิดข้อผิดพลาดในการโหลดโมเดล Embedding: {e}")
    print("  กรุณาตรวจสอบว่าติดตั้ง sentence-transformers และ langchain-huggingface แล้ว")
    sys.exit()

if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)  # Clean up old DB if exists

# Batch processing to avoid memory issues
batch_size = 100
total_chunks = len(all_chunks)

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH,
    collection_name="thairath_news"
)

print(f"Saving to ChromaDB in batches of {batch_size}...")
for i in tqdm(range(0, total_chunks, batch_size)):
    batch = all_chunks[i : i + batch_size]
    vector_store.add_documents(documents=batch)

print(f"\nVector Store saved successfully at: {CHROMA_PATH}")