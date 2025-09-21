import os
import re
import json
from collections import defaultdict

# -------- Phase 1: Indexing the Corpus -------- #

class SearchEngineIndex:
    def __init__(self):
        self.paragraphs = {}        # paragraph_id -> paragraph text
        self.inverted_index = defaultdict(list)  # token -> list of (paragraph_id, positions)
        self.vocab = set()          # unique tokens for auto-correct

    def break_into_paragraphs(self, folder_path):
        """Read all text files and split into paragraphs."""
        pid = 0
        for file in os.listdir(folder_path):
            if file.endswith(".txt"):
                with open(os.path.join(folder_path, file), "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                    paragraphs = text.split("\n\n")  # split on blank lines
                    for para in paragraphs:
                        clean_para = para.strip()
                        if len(clean_para) > 50:  # ignore very short paras
                            self.paragraphs[pid] = {"text": clean_para, "source": file}
                            pid += 1
        print(f"Indexed {len(self.paragraphs)} paragraphs from {len(os.listdir(folder_path))} files.")

    def tokenize(self, text):
        """Simple tokenizer (can later extend with BPE)."""
        tokens = re.findall(r"\b\w+\b", text.lower())
        return tokens

    def build_inverted_index(self):
        """Build an inverted index mapping token -> paragraph IDs and positions."""
        for pid, pdata in self.paragraphs.items():
            tokens = self.tokenize(pdata["text"])
            self.vocab.update(tokens)
            for pos, token in enumerate(tokens):
                self.inverted_index[token].append((pid, pos))

    def save_index(self, path="index.json"):
        """Save paragraphs, index, and vocab to disk."""
        data = {
            "paragraphs": self.paragraphs,
            "inverted_index": {k: v for k, v in self.inverted_index.items()},
            "vocab": list(self.vocab)   # FIX: convert set to list
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        print(f"Index saved to {path}")

# ------------------ Usage ------------------ #
if __name__ == "__main__":
    engine = SearchEngineIndex()
    engine.break_into_paragraphs("corpus")   # folder with .txt files
    engine.build_inverted_index()
    engine.save_index("index.json")