#!/usr/bin/env python3
"""
Script to scrape Eurobarometer dataset URLs with parallel workers.

Usage:
    python run_eurobarometer_scraper.py --start 1 --end 5000
    python run_eurobarometer_scraper.py --start 3000 --end 4000 --workers 8
"""

import argparse

from loguru import logger
from src.surveys.eurobarometer_scraper import EurobarometerScraper


def main():
    parser = argparse.ArgumentParser(
        description="Scrape Eurobarometer dataset URLs from europa.eu"
    )
    parser.add_argument(
        "--start", "-s", type=int, default=1, help="Starting survey index (default: 1)"
    )
    parser.add_argument(
        "--end",
        "-e",
        type=int,
        required=True,
        help="Ending survey index (required)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="survey_dataset_urls.json",
        help="Output JSON file (default: survey_dataset_urls.json)",
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=10,
        help="Timeout in seconds per page (default: 10)",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=4,
        help="Number of parallel browser instances (default: 4)",
    )

    args = parser.parse_args()

    scraper = EurobarometerScraper(
        output_file=args.output,
        timeout=args.timeout,
        n_workers=args.workers,
    )

    results = scraper.scrape_range(start_idx=args.start, end_idx=args.end)

    logger.success(
        f"\nDone! {len(results)} URLs saved to survey_dataset_urls_final.json"
    )


if __name__ == "__main__":
    main()
