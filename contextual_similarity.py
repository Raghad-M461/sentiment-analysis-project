from sentence_transformers import SentenceTransformer, util

PAIRS = [
    ("good", "not good", "Pair 1 — single word vs negated", 0.9116),
    ("the service was great", "the service was not great", "Pair 2 — full sentence flipped", 0.9482),
    ("I am happy with this product", "I am not happy with this product", "Pair 3 — longer sentence with negation", 0.9799),
]

def run():
    print("Loading all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("\nSimilarity Results")
    print("=" * 50)

    for sent_a, sent_b, label, glove_sim in PAIRS:
        embedding_a = model.encode(sent_a, convert_to_tensor=True)
        embedding_b = model.encode(sent_b, convert_to_tensor=True)

        similarity = float(util.cos_sim(embedding_a, embedding_b))

        print(f"\n{label}")
        print(f"  A: {sent_a}")
        print(f"  B: {sent_b}")
        print(f"  GloVe (last week): {glove_sim:.4f}")
        print(f"  MiniLM (now):      {similarity:.4f}")

if __name__ == "__main__":
    run()
