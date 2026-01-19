#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import requests

load_dotenv()

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PERSIST_DIR = REPO_ROOT / "indexing" / "chroma"
DEFAULT_COLLECTION = "knowledge_base"
DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"


@dataclass
class RetrievedChunk:
    id: str
    text: str
    title: str
    source_path: str
    similarity: float
    chunk_id: int


class RAGBot:
    # Few-shot examples using masked terms from terms_map.json
    FEW_SHOT_EXAMPLES = [
        {
            "question": "Who are the Cavilles?",
            "answer": """Let me analyze this step by step:
1. The context mentions Cavilles as elite warriors
2. They serve as the personal guard of Some-ancient-turk
3. They are known for their exceptional combat abilities

Answer: The Cavilles are the elite personal guard of Some-ancient-turk, renowned for their exceptional combat prowess and unwavering loyalty."""
        },
        {
            "question": "What is the MKAD?",
            "answer": """Let me analyze this step by step:
1. The context describes MKAD as a parallel dimension
2. It is used for faster-than-light travel
3. It is also the source of dangerous psychic phenomena

Answer: The MKAD is a parallel dimension of psychic energy that enables faster-than-light travel but is also extremely dangerous due to its chaotic nature."""
        },
        {
            "question": "What happened during the Mavrodi Hoax?",
            "answer": """Let me analyze this step by step:
1. The Mavrodi Hoax was a great civil war
2. It was led by Mavrodi Sergei against Some-ancient-turk
3. It resulted in massive destruction and the current state of Moscow

Answer: The Mavrodi Hoax was a devastating civil war where Mavrodi Sergei betrayed Some-ancient-turk, leading to widespread destruction and fundamentally changing Moscow forever."""
        }
    ]
    
    # System prompt with Chain-of-Thought instructions
    SYSTEM_PROMPT = """You are a knowledgeable assistant that answers questions based on the provided context.

IMPORTANT INSTRUCTIONS:
1. Always reason step-by-step before giving your final answer
2. Use numbered steps to show your thinking process
3. Only use information from the provided context
4. If the context doesn't contain enough information, say "I don't have enough information to answer this question"
5. Be concise but thorough in your reasoning
6. End with a clear "Answer:" section

Format your response like this:
Let me analyze this step by step:
1. [First observation from context]
2. [Second observation]
3. [Conclusion based on observations]

Answer: [Your final answer based on the reasoning above]"""

    def __init__(
        self,
        persist_dir: Path = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION,
        embed_model: str = DEFAULT_EMBED_MODEL,
        llm_model: str = "llama-2-7b-chat",
        top_k: int = 5,
        similarity_threshold: float = 0.3,
        use_few_shot: bool = True,
        use_cot: bool = True,
        verbose: bool = False
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embed_model_name = embed_model
        self.llm_model = llm_model
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.use_few_shot = use_few_shot
        self.use_cot = use_cot
        self.verbose = verbose
        
        # Initialize components
        self._init_chroma()
        self._init_embedder()
        self._init_llm()
    
    def _init_chroma(self) -> None:
        if self.verbose:
            print(f"[INFO] Loading ChromaDB from {self.persist_dir}")
        
        self.chroma_client = chromadb.Client(Settings(
            persist_directory=str(self.persist_dir),
            is_persistent=True,
            anonymized_telemetry=False
        ))
        
        try:
            self.collection = self.chroma_client.get_collection(self.collection_name)
            if self.verbose:
                print(f"[INFO] Loaded collection '{self.collection_name}' with {self.collection.count()} chunks")
        except Exception as e:
            raise RuntimeError(f"Failed to load collection '{self.collection_name}': {e}")
    
    def _init_embedder(self) -> None:
        if self.verbose:
            print(f"[INFO] Loading embedding model: {self.embed_model_name}")
        self.embedder = SentenceTransformer(self.embed_model_name)
    
    def _init_llm(self) -> None:
        if OpenAI is None:
            raise RuntimeError("OpenAI SDK not installed. Run: pip install openai")
        
        api_key = os.getenv("OPENAI_API_KEY", "sk-local")
        base_url = os.getenv("OPENAI_API_BASE", "http://localhost:1234/v1")
        
        if self.verbose:
            print(f"[INFO] LLM API base: {base_url}")
            print(f"[INFO] LLM model: {self.llm_model}")
        
        self.llm_client = OpenAI(api_key=api_key, base_url=base_url)
    
    def embed_query(self, query: str) -> List[float]:
        vector = self.embedder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]
        return vector.astype("float32").tolist()
    
    def retrieve(self, query: str) -> List[RetrievedChunk]:
        embedding = self.embed_query(query)
        
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        chunks = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        
        if self.verbose:
            print(f"[DEBUG] Raw distances: {dists[:5] if dists else 'none'}")
        
        for i in range(len(ids)):
            distance = dists[i] if i < len(dists) else 2.0
            similarity = max(0, 1 - (distance / 2))
            
            if self.verbose:
                print(f"[DEBUG] Chunk {i}: distance={distance:.4f}, similarity={similarity:.4f}")
            
            if similarity < self.similarity_threshold:
                continue
            
            meta = metas[i] if i < len(metas) else {}
            chunks.append(RetrievedChunk(
                id=ids[i],
                text=docs[i] if i < len(docs) else "",
                title=meta.get("title", "Unknown"),
                source_path=meta.get("source_path", ""),
                similarity=similarity,
                chunk_id=meta.get("chunk_id", 0)
            ))
        
        return chunks
    
    def build_context(self, chunks: List[RetrievedChunk], max_chars: int = 2000) -> str:
        if not chunks:
            return "No relevant information found in the knowledge base."
        
        context_parts = []
        total_chars = 0
        
        for i, chunk in enumerate(chunks, 1):
            chunk_text = f"[Source {i}: {chunk.title}]\n{chunk.text}"
            
            if total_chars + len(chunk_text) > max_chars:
                remaining = max_chars - total_chars
                if remaining > 200:
                    chunk_text = chunk_text[:remaining] + "..."
                    context_parts.append(chunk_text)
                break
            
            context_parts.append(chunk_text)
            total_chars += len(chunk_text)
        
        return "\n\n".join(context_parts)
    
    def build_few_shot_section(self) -> str:
        if not self.use_few_shot:
            return ""
        
        examples = []
        for ex in self.FEW_SHOT_EXAMPLES:
            examples.append(f"Question: {ex['question']}\n{ex['answer']}")
        
        return "Here are examples of how to answer:\n\n" + "\n\n---\n\n".join(examples)
    
    def build_prompt(self, query: str, chunks: List[RetrievedChunk]) -> List[Dict[str, str]]:
        context = self.build_context(chunks)
        few_shot = self.build_few_shot_section()
        
        system_content = self.SYSTEM_PROMPT if self.use_cot else "You are a helpful assistant. Answer based on the provided context only."
        
        user_parts = []
        
        if few_shot:
            user_parts.append(few_shot)
        
        user_parts.append(f"Context from knowledge base:\n{context}")
        user_parts.append(f"Question: {query}")
        
        user_content = "\n\n".join(user_parts)
        
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
    
    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        base_url = os.getenv("OPENAI_API_BASE", "http://localhost:1234/v1").rstrip("/v1").rstrip("/")
        errors = []
        
        prompt = f"""### SYSTEM:
{messages[0]['content'] if messages else ''}

### USER:
{messages[1]['content'] if len(messages) > 1 else ''}

### ASSISTANT:
"""
        
        try:
            if self.verbose:
                print("[DEBUG] Trying direct HTTP chat completions...")
            
            url = f"{base_url}/v1/chat/completions"
            payload = {
                "model": self.llm_model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1024
            }
            
            resp = requests.post(url, json=payload, timeout=120)
            if self.verbose:
                print(f"[DEBUG] HTTP status: {resp.status_code}")
                print(f"[DEBUG] HTTP response: {resp.text[:500] if resp.text else 'empty'}")
            
            if resp.status_code == 200:
                data = resp.json()
                if "choices" in data and len(data["choices"]) > 0:
                    choice = data["choices"][0]
                    if "message" in choice and "content" in choice["message"]:
                        return choice["message"]["content"]
                    elif "text" in choice:
                        return choice["text"]
                errors.append(f"chat/completions: no valid content in response")
            else:
                errors.append(f"chat/completions: HTTP {resp.status_code}")
        except Exception as e:
            errors.append(f"chat/completions: {e}")
        
        try:
            if self.verbose:
                print("[DEBUG] Trying direct HTTP completions...")
            
            url = f"{base_url}/v1/completions"
            payload = {
                "model": self.llm_model,
                "prompt": prompt,
                "temperature": 0.7,
                "max_tokens": 1024,
                "stop": ["### USER:", "### SYSTEM:"]
            }
            
            resp = requests.post(url, json=payload, timeout=120)
            if self.verbose:
                print(f"[DEBUG] HTTP status: {resp.status_code}")
                print(f"[DEBUG] HTTP response: {resp.text[:500] if resp.text else 'empty'}")
            
            if resp.status_code == 200:
                data = resp.json()
                if "choices" in data and len(data["choices"]) > 0:
                    text = data["choices"][0].get("text", "")
                    if text:
                        return text.strip()
                errors.append(f"completions: no valid text in response")
            else:
                errors.append(f"completions: HTTP {resp.status_code}")
        except Exception as e:
            errors.append(f"completions: {e}")
        
        try:
            if self.verbose:
                print("[DEBUG] Trying Ollama-style /api/generate...")
            
            url = f"{base_url}/api/generate"
            payload = {
                "model": self.llm_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 1024
                }
            }
            
            resp = requests.post(url, json=payload, timeout=120)
            if self.verbose:
                print(f"[DEBUG] HTTP status: {resp.status_code}")
                print(f"[DEBUG] HTTP response: {resp.text[:500] if resp.text else 'empty'}")
            
            if resp.status_code == 200:
                data = resp.json()
                if "response" in data:
                    return data["response"].strip()
                errors.append(f"api/generate: no 'response' field")
            else:
                errors.append(f"api/generate: HTTP {resp.status_code}")
        except Exception as e:
            errors.append(f"api/generate: {e}")
        
        try:
            if self.verbose:
                print("[DEBUG] Trying Ollama-style /api/chat...")
            
            url = f"{base_url}/api/chat"
            payload = {
                "model": self.llm_model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 1024
                }
            }
            
            resp = requests.post(url, json=payload, timeout=120)
            if self.verbose:
                print(f"[DEBUG] HTTP status: {resp.status_code}")
                print(f"[DEBUG] HTTP response: {resp.text[:500] if resp.text else 'empty'}")
            
            if resp.status_code == 200:
                data = resp.json()
                if "message" in data and "content" in data["message"]:
                    return data["message"]["content"].strip()
                errors.append(f"api/chat: no valid content")
            else:
                errors.append(f"api/chat: HTTP {resp.status_code}")
        except Exception as e:
            errors.append(f"api/chat: {e}")
        
        error_msg = "LLM API Error - all methods failed:\n" + "\n".join(f"  - {e}" for e in errors)
        error_msg += f"\n\nBase URL: {base_url}"
        error_msg += "\n\nPlease check:\n1. LLM server is running\n2. OPENAI_API_BASE is correct\n3. Model name is correct"
        raise RuntimeError(error_msg)
    
    def answer(self, query: str) -> Dict:
        if not query.strip():
            return {
                "answer": "Please provide a question.",
                "sources": [],
                "chunks": []
            }
        
        chunks = self.retrieve(query)
        
        if self.verbose:
            print(f"[INFO] Retrieved {len(chunks)} relevant chunks")
            for c in chunks:
                print(f"  - {c.title} (similarity: {c.similarity:.3f})")
        
        if not chunks:
            return {
                "answer": "I don't have enough information in my knowledge base to answer this question. The query doesn't match any relevant documents.",
                "sources": [],
                "chunks": []
            }
        
        messages = self.build_prompt(query, chunks)
        
        if self.verbose:
            print(f"[INFO] Prompt size: {sum(len(m['content']) for m in messages)} chars")
        
        answer = self.generate_response(messages)
        
        sources = list(set(c.source_path for c in chunks))
        
        return {
            "answer": answer,
            "sources": sources,
            "chunks": [
                {
                    "title": c.title,
                    "similarity": c.similarity,
                    "text_preview": c.text[:200] + "..." if len(c.text) > 200 else c.text
                }
                for c in chunks
            ]
        }
    
    def run_repl(self) -> None:
        print("=" * 60)
        print("RAG Bot Ready")
        print("=" * 60)
        print(f"Knowledge base: {self.collection.count()} chunks")
        print(f"Embedding model: {self.embed_model_name}")
        print(f"LLM model: {self.llm_model}")
        print(f"Few-shot: {'enabled' if self.use_few_shot else 'disabled'}")
        print(f"Chain-of-Thought: {'enabled' if self.use_cot else 'disabled'}")
        print("=" * 60)
        print("Type your question (or 'quit' to exit, 'sources' to show last sources)")
        print()
        
        last_result = None
        
        while True:
            try:
                query = input("Q> ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ('quit', 'exit', 'q'):
                    print("Goodbye!")
                    break
                
                if query.lower() == 'sources' and last_result:
                    print("\nSources from last answer:")
                    for chunk in last_result.get("chunks", []):
                        print(f"  - {chunk['title']} (similarity: {chunk['similarity']:.3f})")
                    print()
                    continue
                
                result = self.answer(query)
                last_result = result
                
                print(f"\nA> {result['answer']}\n")
                
                if result['sources']:
                    print(f"[Sources: {', '.join(result['sources'][:3])}]\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}\n")


def run_demo_dialogues(bot: RAGBot) -> None:
    print("\n" + "=" * 70)
    print("DEMONSTRATION DIALOGUES")
    print("=" * 70)
    
    # Successful dialogues using masked terms from terms_map.json
    successful_queries = [
        "Who is Some-ancient-turk and what is his role?",
        "What is the MVD and what do they do?",
        "Tell me about the Moon Bikers warriors",
        "What happened during the Mavrodi Hoax?",
        "What is the MKAD and why is it dangerous?",
        "Who are the Cavilles and what is their purpose?",
        "Tell me about Saint-Petersburg and its importance",
        "What is Moscow and how is it governed?",
        "Who are the Mushrooms and why are they a threat?",
        "What are Cockroaches and where do they come from?",
    ]
    
    print("\n--- SUCCESSFUL DIALOGUES ---\n")
    
    for i, query in enumerate(successful_queries, 1):
        print(f"[Dialogue {i}]")
        print(f"Q: {query}")
        result = bot.answer(query)
        print(f"A: {result['answer']}")
        if result['sources']:
            print(f"[Sources: {', '.join(result['sources'][:2])}]")
        print("-" * 50)
        print()
    
    # "I don't know" cases - queries about things not in the knowledge base
    unknown_queries = [
        "What is the recipe for pizza?",
        "Who won the World Cup in 2022?",
        "What is the capital of France?",
    ]
    
    print("\n--- 'I DON'T KNOW' CASES ---\n")
    
    for i, query in enumerate(unknown_queries, 1):
        print(f"[Unknown Case {i}]")
        print(f"Q: {query}")
        result = bot.answer(query)
        print(f"A: {result['answer']}")
        print("-" * 50)
        print()


def main():
    parser = argparse.ArgumentParser(
        description="RAG Bot with Few-shot and Chain-of-Thought",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--persist-dir",
        default=str(DEFAULT_PERSIST_DIR),
        help="ChromaDB persistence directory"
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help="ChromaDB collection name"
    )
    parser.add_argument(
        "--embed-model",
        default=DEFAULT_EMBED_MODEL,
        help="Sentence transformer model for embeddings"
    )
    parser.add_argument(
        "--llm-model",
        default="llama-2-7b-chat",
        help="LLM model name"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.3,
        help="Minimum similarity score for retrieved chunks"
    )
    parser.add_argument(
        "--no-few-shot",
        action="store_true",
        help="Disable few-shot examples"
    )
    parser.add_argument(
        "--no-cot",
        action="store_true",
        help="Disable Chain-of-Thought prompting"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demonstration dialogues instead of REPL"
    )
    
    args = parser.parse_args()
    
    bot = RAGBot(
        persist_dir=Path(args.persist_dir),
        collection_name=args.collection,
        embed_model=args.embed_model,
        llm_model=args.llm_model,
        top_k=args.top_k,
        similarity_threshold=args.similarity_threshold,
        use_few_shot=not args.no_few_shot,
        use_cot=not args.no_cot,
        verbose=args.verbose
    )
    
    if args.demo:
        run_demo_dialogues(bot)
    else:
        bot.run_repl()


if __name__ == "__main__":
    main()
