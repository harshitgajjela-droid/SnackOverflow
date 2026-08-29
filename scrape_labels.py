import os
from icrawler.builtin import BingImageCrawler, GoogleImageCrawler #

def scrape_product_labels():
    print("========================================")
    print("   CODE WARRIORS - DATASET SCRAPER")
    print("========================================\n")

    # Define where the images will be saved
    dataset_dir = 'dataset/images/train'
    os.makedirs(dataset_dir, exist_ok=True)

    # Keywords designed to find the back of Indian product packages
    search_queries = [
        "FSSAI product label back",
        "Indian FMCG product packaging back",
        "packaged food nutritional label India",
        "FSSAI veg non veg dot packaging",
        "ISI mark product label"
    ]

    images_per_query = 50  # Adjust this number based on how many you want

    print(f"Target Directory: {dataset_dir}")
    print(f"Running {len(search_queries)} queries for {images_per_query} images each...\n")

    # Using Bing Image Crawler (often more permissive than Google for automated scraping)
    crawler = BingImageCrawler(
        downloader_threads=4, # Speeds up downloads
        storage={'root_dir': dataset_dir} #
    )

    for query in search_queries:
        print(f"\n---> Scraping images for: '{query}'")
        
        # You can add filters if needed, e.g., 'size': 'large'
        crawler.crawl(
            keyword=query,
            max_num=images_per_query #
        )

    print("\n========================================")
    print("   SCRAPING COMPLETE!")
    print(f"   Check your '{dataset_dir}' folder.")
    print("========================================")

if __name__ == "__main__":
    scrape_product_labels()