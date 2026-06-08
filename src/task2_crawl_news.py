from personal_submission.vu_anh.src.task2_crawl_news import ARTICLE_URLS, crawl_article, crawl_all

if __name__ == "__main__":
    import asyncio
    asyncio.run(crawl_all())
