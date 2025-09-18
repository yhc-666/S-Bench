"""Unified dataset loader for all benchmarks."""

from typing import Dict, List, Any
import datasets
from .base_dataset import BaseDataset


class BenchmarkDataset(BaseDataset):
    """Load any of the benchmark datasets."""

    def load(self) -> List[Dict[str, Any]]:
        """Load dataset from HuggingFace or cache."""
        # Try cache first
        cached_data = self.load_cache()
        if cached_data:
            print(f"Loading {self.subset} from cache...")
            if self.test_size > 0:
                return cached_data[:self.test_size]
            return cached_data

        # Load from HuggingFace
        print(f"Loading {self.subset} from HuggingFace...")
        dataset = datasets.load_dataset(self.source, self.subset, split='test')

        # Process dataset
        processed_data = []
        for idx, item in enumerate(dataset):
            # Handle different dataset formats
            question = item.get('question', item.get('query', ''))
            question = self.format_question(question)

            # Get answers - handle different formats
            answers = item.get('golden_answers', item.get('answer', []))
            if isinstance(answers, str):
                answers = [answers]

            processed_item = {
                'id': f"{self.subset}_{idx}",
                'question': question,
                'answers': answers,
                'metadata': {
                    'dataset': self.subset,
                    'index': idx
                }
            }
            processed_data.append(processed_item)

            # Apply test size limit
            if self.test_size > 0 and len(processed_data) >= self.test_size:
                break

        # Save to cache
        self.save_cache(processed_data)

        return processed_data