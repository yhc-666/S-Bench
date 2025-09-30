#!/usr/bin/env python3
"""Download MINTQA datasets from GitHub."""

import os
import sys
import json
import argparse
import requests
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MINTQADownloader:
    """Download and process MINTQA datasets."""

    GITHUB_BASE_URL = "https://raw.githubusercontent.com/probe2/multi-hop/main/"
    DATASETS = {
        'mintqa_pop': 'MINTQA-POP.json',
        'mintqa_ti': 'MINTQA-TI.json'
    }

    def __init__(self, cache_dir: str = None):
        """Initialize downloader."""
        if cache_dir is None:
            cache_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'data'
            )
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_json(self, url: str) -> Dict:
        """Download JSON data from URL."""
        print(f"Downloading from {url}...")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error downloading data: {e}")
            raise

    def process_mintqa_data(self, data: Any, dataset_name: str) -> List[Dict]:
        """Process MINTQA data into standard format."""
        processed_data = []

        # Handle different possible formats
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            # Try to find the data array in the dict
            if 'data' in data:
                items = data['data']
            elif 'questions' in data:
                items = data['questions']
            else:
                # Assume the dict itself contains numbered items
                items = list(data.values())
        else:
            raise ValueError(f"Unexpected data format: {type(data)}")

        for idx, item in enumerate(tqdm(items, desc=f"Processing {dataset_name}")):
            # Extract question
            if isinstance(item, dict):
                # Try different field names for question
                question = item.get('question',
                          item.get('query',
                          item.get('text', '')))

                # Extract answers - try different field names
                answers = item.get('answer',
                         item.get('answers',
                         item.get('golden_answers', [])))

                # Handle if answers is a single string
                if isinstance(answers, str):
                    answers = [answers]
                elif not isinstance(answers, list):
                    answers = [str(answers)] if answers else []

                # Also check for sub-questions if available
                sub_questions = item.get('sub_questions', [])
                sub_answers = item.get('sub_answers', [])

            else:
                # If item is not a dict, try to parse it
                question = str(item)
                answers = []
                sub_questions = []
                sub_answers = []

            # Ensure question ends with question mark
            question = question.strip()
            if question and not question.endswith('?'):
                question += '?'

            # Create processed item in standard format
            processed_item = {
                'id': f"{dataset_name}_{idx}",
                'question': question,
                'answers': answers,
                'metadata': {
                    'dataset': dataset_name,
                    'index': idx
                }
            }

            # Add sub-questions to metadata if available
            if sub_questions:
                processed_item['metadata']['sub_questions'] = sub_questions
            if sub_answers:
                processed_item['metadata']['sub_answers'] = sub_answers

            processed_data.append(processed_item)

        return processed_data

    def download_dataset(self, dataset_name: str, force: bool = False) -> bool:
        """Download and process a single MINTQA dataset."""
        if dataset_name not in self.DATASETS:
            print(f"Error: Unknown dataset {dataset_name}")
            print(f"Available datasets: {', '.join(self.DATASETS.keys())}")
            return False

        # Create dataset directory
        dataset_dir = self.cache_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Check if already cached
        cache_file = dataset_dir / "data.jsonl"
        if cache_file.exists() and not force:
            print(f"✓ {dataset_name} already cached at {cache_file}")
            return True

        # Download data
        url = self.GITHUB_BASE_URL + self.DATASETS[dataset_name]
        try:
            data = self.download_json(url)
        except Exception as e:
            print(f"✗ Failed to download {dataset_name}: {e}")
            return False

        # Process data
        try:
            processed_data = self.process_mintqa_data(data, dataset_name)

            # Save to cache in JSONL format
            with open(cache_file, 'w') as f:
                for item in processed_data:
                    f.write(json.dumps(item) + '\n')

            print(f"✓ {dataset_name}: {len(processed_data)} examples saved to {cache_file}")
            return True

        except Exception as e:
            print(f"✗ Failed to process {dataset_name}: {e}")
            return False

    def download_all(self, force: bool = False) -> Dict[str, bool]:
        """Download all MINTQA datasets."""
        results = {}

        print("\n" + "="*60)
        print("DOWNLOADING MINTQA DATASETS")
        print("="*60)

        for dataset_name in self.DATASETS.keys():
            results[dataset_name] = self.download_dataset(dataset_name, force)

        # Summary
        print("\n" + "="*60)
        print("DOWNLOAD SUMMARY")
        print("="*60)

        successful = sum(1 for v in results.values() if v)
        failed = len(results) - successful

        print(f"✓ Successful: {successful}")
        print(f"✗ Failed: {failed}")

        if failed > 0:
            print("\nFailed datasets:")
            for name, success in results.items():
                if not success:
                    print(f"  - {name}")

        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download MINTQA datasets from GitHub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all MINTQA datasets
  python download_mintqa.py

  # Download specific dataset
  python download_mintqa.py --dataset mintqa_pop

  # Force re-download
  python download_mintqa.py --force

  # Use custom cache directory
  python download_mintqa.py --cache-dir ./my_cache
        """
    )

    parser.add_argument(
        '--dataset',
        choices=['mintqa_pop', 'mintqa_ti'],
        help='Specific dataset to download (default: all)'
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        help='Directory to cache datasets'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if cached'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available MINTQA datasets'
    )

    args = parser.parse_args()

    # Initialize downloader
    downloader = MINTQADownloader(cache_dir=args.cache_dir)

    # List datasets if requested
    if args.list:
        print("\nAvailable MINTQA datasets:")
        print("-" * 40)
        for name, filename in downloader.DATASETS.items():
            print(f"  {name:15} -> {filename}")
        return

    # Download datasets
    if args.dataset:
        # Download single dataset
        success = downloader.download_dataset(args.dataset, args.force)
        if not success:
            sys.exit(1)
    else:
        # Download all datasets
        results = downloader.download_all(args.force)
        if any(not v for v in results.values()):
            sys.exit(1)


if __name__ == "__main__":
    main()