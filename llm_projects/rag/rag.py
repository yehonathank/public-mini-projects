import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class RAG_Setup:
    def __init__(self, doc_paths: list[str], k_nearest_chunks: int, sentences_per_chunk: int):
        self.doc_paths = doc_paths
        self.k_nearest_chunks = k_nearest_chunks
        self.sentences_per_chunk = sentences_per_chunk

        self.chunks = []
        self.embeddings = []
        self.index = None
        self.model = SentenceTransformer('all-MiniLM-L6-v2') # lightweight and fast

    def run_setup(self):
        self.load_or_chunk_docs()
        self.embed_chunks()

    def load_or_chunk_docs(self):
        """Load chunks from folder if exists, otherwise chunk and save."""
        for path in self.doc_paths:
            base = os.path.splitext(os.path.basename(path))[0]
            chunk_folder = os.path.join(os.path.dirname(path), f"{base}_chunks")
            if os.path.exists(chunk_folder) and os.path.isdir(chunk_folder):
                # Load chunks from files
                chunk_files = sorted([
                    f for f in os.listdir(chunk_folder)
                    if f.startswith('chunk') and f.endswith('.txt')
                ], key=lambda x: int(x.replace('chunk', '').replace('.txt', '')))
                for chunk_file in chunk_files:
                    chunk_path = os.path.join(chunk_folder, chunk_file)
                    with open(chunk_path, 'r', encoding='utf-8') as cf:
                        chunk = cf.read().strip()
                        if chunk:
                            self.chunks.append(chunk)
            else:
                # Chunk and save as before
                if not os.path.exists(path):
                    continue
                with open(path, 'r', encoding='utf-8') as f:
                    text = f.read()
                # Split text into sentences by period, preserving periods
                raw_sentences = text.split('.')
                sentences = [s.strip() for s in raw_sentences if s.strip()]
                sentences = [s + '.' for s in sentences]  # add period back
                chunks = []
                for i in range(0, len(sentences), self.sentences_per_chunk):
                    chunk = ' '.join(sentences[i:i+self.sentences_per_chunk]).strip()
                    if chunk:
                        chunks.append(chunk)
                os.makedirs(chunk_folder, exist_ok=True)
                for i, chunk in enumerate(chunks):
                    cleaned = chunk.strip()
                    if cleaned:
                        self.chunks.append(cleaned)
                        chunk_path = os.path.join(chunk_folder, f"chunk{i+1}.txt")
                        with open(chunk_path, 'w', encoding='utf-8') as cf:
                            cf.write(cleaned)

    def embed_chunks(self):
        """PLAN:
        1. create a faiss vector DB
        """
        self.embeddings = self.model.encode(self.chunks, convert_to_numpy=True)
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)

    def retrieve_k_context(self, user_input: str) -> list[str]:
        """PLAN: 
        1. embed user_input and compare to faiss vector DB
        2. grab k nearest chunks from faiss
        3. return a k nearest context
        """
        #checks
        if self.index is None:
            self.run_setup()

        query_embedding = self.model.encode([user_input], convert_to_numpy=True).astype("float32")
        if not isinstance(self.index, faiss.Index):
            raise TypeError(f"Expected faiss.Index, got {type(self.index)}")
        
        distances, indices = self.index.search(query_embedding, self.k_nearest_chunks)
        print(f"DEBUG: distances = {distances}")
        print(f"DEBUG: indices = {indices}")
        return [self.chunks[i] for i in indices[0]]