from personal_submission.vu_anh.src.task9_retrieval_pipeline import retrieve

if __name__ == "__main__":
    results = retrieve("cai nghiện ma túy", top_k=3)
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
