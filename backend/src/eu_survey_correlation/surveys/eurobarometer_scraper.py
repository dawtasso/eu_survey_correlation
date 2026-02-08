import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from loguru import logger
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm


class EurobarometerScraper:
    """Scrapes dataset URLs from Eurobarometer survey pages."""

    def __init__(
        self,
        output_file: str = "dataset_urls.json",
        timeout: int = 10,
        max_consecutive_failures: int = 5,
        n_workers: int = 4,
    ):
        """
        Args:
            output_file: Path to save the JSON results
            timeout: Max seconds to wait for each page to load
            max_consecutive_failures: Stop after this many consecutive failures
            n_workers: Number of parallel browser instances
        """
        self.output_file = Path(output_file)
        self.timeout = timeout
        self.max_consecutive_failures = max_consecutive_failures
        self.n_workers = n_workers
        self.results = {}
        self.base_url = "https://europa.eu/eurobarometer/surveys/detail/"
        self._lock = Lock()

        # Load existing results if file exists
        if self.output_file.exists():
            with open(self.output_file, "r", encoding="utf-8") as f:
                self.results = json.load(f)
            logger.info(
                f"Loaded {len(self.results)} existing entries from {self.output_file}"
            )

    def _setup_driver(self):
        """Setup headless Chrome browser."""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument(
            "--blink-settings=imagesEnabled=false"
        )  # Disable images
        return webdriver.Chrome(options=chrome_options)

    def _scrape_with_driver(self, driver, idx: int) -> tuple[int, str | None]:
        """Scrape a single index using an existing driver."""
        dataset_url = None
        try:
            url = f"{self.base_url}{idx}"
            driver.get(url)

            wait = WebDriverWait(driver, self.timeout)
            element = wait.until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, 'a[href*="data.europa.eu/euodp/en/data/dataset"]')
                )
            )
            dataset_url = element.get_attribute("href")
        except Exception:
            pass

        return idx, dataset_url

    def _add_result(self, idx: int, url: str):
        """Thread-safe method to add a result and save immediately."""
        with self._lock:
            self.results[str(idx)] = url
            self._save()

    def _worker(self, indices: list[int], pbar: tqdm):
        """Worker function that processes a batch of indices with a single driver."""
        driver = self._setup_driver()

        try:
            for idx in indices:
                idx, dataset_url = self._scrape_with_driver(driver, idx)
                if dataset_url:
                    self._add_result(idx, dataset_url)  # Save immediately
                pbar.update(1)
        finally:
            driver.quit()

    def scrape_range(self, start_idx: int, end_idx: int):
        """
        Scrape dataset URLs for a range of indices using parallel workers.

        Args:
            start_idx: Starting survey index
            end_idx: Ending survey index
        """
        # Filter out already scraped indices
        indices_to_scrape = [
            idx for idx in range(start_idx, end_idx) if str(idx) not in self.results
        ]

        if not indices_to_scrape:
            logger.info("All indices already scraped!")
            return self.results

        logger.info(
            f"Scraping {len(indices_to_scrape)} indices with {self.n_workers} workers..."
        )

        # Split indices into chunks for each worker
        chunk_size = max(1, len(indices_to_scrape) // self.n_workers)
        chunks = [
            indices_to_scrape[i : i + chunk_size]
            for i in range(0, len(indices_to_scrape), chunk_size)
        ]

        with tqdm(total=len(indices_to_scrape), desc="Scraping surveys") as pbar:
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                futures = [
                    executor.submit(self._worker, chunk, pbar) for chunk in chunks
                ]
                # Wait for all workers to complete
                for future in as_completed(futures):
                    future.result()  # Raise any exceptions

        logger.success(f"Scraping complete. Total URLs collected: {len(self.results)}")
        return self.results

    def _save(self):
        """Save results to JSON file."""
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)

    def get_results(self) -> dict:
        """Return all collected results."""
        return self.results


if __name__ == "__main__":
    # Example usage
    scraper = EurobarometerScraper(
        output_file="dataset_urls.json",
        timeout=10,
        max_consecutive_failures=5,
        n_workers=4,  # 4 parallel browsers
    )

    # Scrape a range with parallel workers
    scraper.scrape_range(start_idx=1, end_idx=5000)
