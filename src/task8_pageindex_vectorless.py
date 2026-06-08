from personal_submission.vu_anh.src.task8_pageindex_vectorless import pageindex_search, upload_documents

if __name__ == "__main__":
    results = pageindex_search("ma tuý", top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
