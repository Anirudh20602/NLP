import os
import re
import json
import sentencepiece as spm
from collections import Counter, defaultdict
from pathlib import Path

# PHASE 1: Build Inverted Index with BPE Tokenization
class SearchEngine:
    def __init__(self, corpus_dir, index_file="inverted_index.json"):
        self.corpus_dir = corpus_dir
        self.index_file = index_file
        self.inverted_index = {}
        self.vocabulary = set()
        self.paragraphs = {}  # {para_id: {"text": str, "book": str}}
        self.sp = None

    def train_bpe(self, model_prefix="bpe_model", vocab_size=5000):
        """Train BPE model on the corpus"""
        combined_file = "corpus_combined.txt"
        with open(combined_file, "w", encoding="utf-8") as f:
            for book in os.listdir(self.corpus_dir):
                if book.endswith(".txt"):
                    with open(os.path.join(self.corpus_dir, book), "r", encoding="utf-8", errors="ignore") as infile:
                        f.write(infile.read() + "\n")
        
        spm.SentencePieceTrainer.train(
            input=combined_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="bpe"
        )
        self.sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")

    def tokenize(self, text):
        """Tokenize using BPE model"""
        if not self.sp:
            raise ValueError("BPE model not trained/loaded yet.")
        return self.sp.encode(text, out_type=str)

    def build_index(self):
        """Build inverted index for all books"""
        para_id = 0
        for book in os.listdir(self.corpus_dir):
            if book.endswith(".txt"):
                with open(os.path.join(self.corpus_dir, book), "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    paragraphs = re.split(r"\n\s*\n", content)  # split by blank lines

                    for p in paragraphs:
                        p = p.strip()
                        if not p:
                            continue
                        self.paragraphs[para_id] = {"text": p, "book": book}
                        tokens = self.tokenize(p)
                        for token in tokens:
                            self.inverted_index.setdefault(token, []).append(para_id)
                            self.vocabulary.add(token)
                        para_id += 1

        # Save index
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump({
                "inverted_index": self.inverted_index,
                "paragraphs": self.paragraphs,
                "vocabulary": list(self.vocabulary)
            }, f, indent=2)

        print(f"Index built and saved to {self.index_file}")

if __name__ == "__main__":
    corpus_path = r"C:\PROJECTS\NLP\corpus"  # change path
    engine = SearchEngine(corpus_path)
    engine.train_bpe()         # train BPE model
    engine.build_index()       # build and save index

print("Total paragraphs indexed:", len(engine.paragraphs))
print("Total vocabulary size:", len(engine.vocabulary))

# PHASE 2: Search Functionality
class SearchEngine(SearchEngine):  # extend the previous class definition
    def load_index(self):
        """Load BPE model + inverted index + paragraphs + vocabulary."""
        if not hasattr(self, 'spm_model'):
            self.spm_model = "bpe_model.model"
        if not Path(self.spm_model).exists():
            raise FileNotFoundError(f"Missing BPE model: {self.spm_model}")
        self.sp = spm.SentencePieceProcessor(model_file=self.spm_model)

        if not Path(self.index_file).exists():
            raise FileNotFoundError(f"Missing index file: {self.index_file}")
        with open(self.index_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.inverted_index = {k: list(set(v)) for k, v in data["inverted_index"].items()}
        # keys in json may be strings; normalize to int keys for paragraphs:
        self.paragraphs = {int(k): v for k, v in data["paragraphs"].items()}
        self.vocabulary = set(data["vocabulary"])
        print(f"Loaded index: {len(self.paragraphs)} paragraphs, vocab {len(self.vocabulary)}")

    def process_query(self, query):
        """Tokenize query and separate in-vocab vs OOV tokens (Phase 3 will handle OOV smarter)."""
        q_tokens = self.tokenize(query)
        in_vocab = [t for t in q_tokens if t in self.vocabulary]
        oov = [t for t in q_tokens if t not in self.vocabulary]
        return q_tokens, in_vocab, oov

    def search_basic(self, query, top_k=10):
        """
        Simple retrieval using TF-style scoring:
          score(pid) = sum over query tokens of tf(token, pid)
        This is enough for Phase 2; we’ll upgrade to tf-idf/cosine in Phase 4.
        """
        _, in_vocab, _ = self.process_query(query)
        if not in_vocab:
            return []

        # collect candidate paragraphs from union of postings
        postings = defaultdict(list)
        for tok in in_vocab:
            for pid in self.inverted_index.get(tok, []):
                postings[pid].append(tok)

        # score by term frequency of query tokens within each paragraph
        scores = []
        for pid, toks in postings.items():
            # Count matches of query tokens in this paragraph
            tf = Counter(toks)
            score = sum(tf.values())
            scores.append((pid, score))

        # rank high → low by score, then by shorter paragraph (tie-breaker)
        scores.sort(key=lambda x: (-x[1], len(self.paragraphs[x[0]]["text"])))
        results = []
        for pid, sc in scores[:top_k]:
            rec = self.paragraphs[pid]
            txt = rec["text"].replace("\n", " ")
            snippet = txt[:220] + ("..." if len(txt) > 220 else "")
            results.append({
                "paragraph_id": pid,
                "book": rec["book"],
                "score": sc,
                "snippet": snippet
            })
        return results

    # Optional utility for quick checks
    def peek_token(self, token, k=3):
        pids = self.inverted_index.get(token, [])
        print(f"Token '{token}' appears in {len(pids)} paragraphs")
        for pid in pids[:k]:
            rec = self.paragraphs[pid]
            text = rec["text"].replace("\n", " ")
            print(f"- [{pid}] {rec['book']}: {text[:160]}...")

# ---------- CLI entry point ----------
if __name__ == "__main__":
    corpus_path = r"C:\PROJECTS\NLP\corpus"  # <-- your corpus folder
    engine = SearchEngine(corpus_path)

    # If index + model are present, skip rebuilding
    have_model = Path(engine.spm_model).exsts()
    have_index = Path(engine.index_file).exists()
    if not (have_model and have_index):
        print("Building index (first run)...")
        engine.train_bpe()      # Phase 1: train BPE
        engine.build_index()    # Phase 1: build inverted index

    # Phase 2: interactive search
    engine.load_index()         # load model + index
    print("\nSearch ready. Type your query (or 'exit' to quit).")
    while True:
        try:
            q = input("Enter search query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if q.lower() in {"exit", "quit"}:
            print("Bye!")
            break
        if not q:
            print("Empty query. Try again.")
            continue

        q_tokens, in_vocab, oov = engine.process_query(q)
        if oov:
            print(f"(Info) OOV tokens (Phase 3 will handle): {oov}")

        results = engine.search_basic(q, top_k=10)
        if not results:
            print("No paragraph found for this query.\n")
            continue

        print("\nTop results:")
        for i, r in enumerate(results, 1):
            print(f"{i:2d}. [pid {r['paragraph_id']}] {r['book']} | score={r['score']}")
            print(f"    {r['snippet']}\n")