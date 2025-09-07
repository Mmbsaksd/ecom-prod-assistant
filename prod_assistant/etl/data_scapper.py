import csv
import time
import re
import os
from bs4 import BeautifulSoup
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

class FlipKartScraper:
    def __init__(self, output_dir="data"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def get_top_reviews(self, product_url, count=2):
        options = uc.ChromeOptions()
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-blink-features=AAutomationControlled")
        driver = uc.Chrome(options=options, use_subprocess=True)

        if not product_url.startswith("http"):
            return "No review found"
        
        try:
            driver.get(product_url)
            time.sleep(4)
            try:
                pass
            except Exception as e:
                print(f"Error occured while closing popup: {e}")
        except Exception:
            reviews = []
        driver.quit()
        return " || ".join(reviews) if reviews else "No review found"
            

    def scrape_flipkart_product(self, query, max_product=1, review_count=2):
        pass

    def save_to_csv(self, data, filename="product_reviews.csv"):
        pass
        

