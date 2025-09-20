#!/usr/bin/env python3
"""Main evaluation script."""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, List
import yaml
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ''))

from src.models.closed_source import OpenAIModel, DeepSeekModel
from src.models.open_source import VLLMModel
from src.datasets.dataset_loader import BenchmarkDataset
from src.search.search_interface import SearchEngine
from src.search.tag_search import TagBasedSearch
from src.search.function_search import FunctionSearchHandler
from src.inference.tag_based_inference import TagBasedInference
from src.inference.function_inference import FunctionInference
from src.metrics.metrics import calculate_metrics


def load_config(config_dir: str) -> Dict[str, Any]:
    """Load all configuration files."""
    configs = {}
    config_files = ['models.yaml', 'datasets.yaml', 'search_engines.yaml', 'prompts.yaml']

    for file in config_files:
        path = os.path.join(config_dir, file)
        with open(path, 'r') as f:
            configs[file.replace('.yaml', '')] = yaml.safe_load(f)

    return configs


def initialize_model(model_name: str, config: Dict) -> Any:
    """Initialize model based on configuration."""
    model_config = config['models']['models'][model_name]

    if model_config['type'] == 'closed_source':
        if 'gpt' in model_name:
            return OpenAIModel(model_config)
        elif 'deepseek' in model_name:
            return DeepSeekModel(model_config)
        elif 'claude' in model_name:
            return OpenAIModel(model_config)  # Claude uses OpenAI-compatible API
    else:
        return VLLMModel(model_config)


def initialize_search(config: Dict, search_method: str) -> Any:
    """Initialize search components."""
    search_config = config['search_engines']['search_engine']

    if search_method == 'tag':
        search_engine = SearchEngine(search_config)
        return TagBasedSearch(search_engine, search_config)
    elif search_method == 'function':
        return FunctionSearchHandler(search_config)
    else:
        raise ValueError(f"Unknown search method: {search_method}")


def get_prompt_config(config: Dict, model_name: str, search_method: str) -> Dict:
    """Get appropriate prompt configuration."""
    prompts = config['prompts']

    if search_method == 'tag':
        return prompts['prompts']['tag_based']
    elif search_method == 'function':
        # Determine model type to select correct prompt variant
        model_config = config['models']['models'][model_name]
        model_type = model_config.get('type', 'closed_source')

        if model_type == 'closed_source':
            return prompts['prompts']['function_based_closed_source']
        else:
            return prompts['prompts']['function_based_open_source']
    else:
        raise ValueError(f"Unknown search method: {search_method}")


def evaluate_single(
    question: str,
    model: Any,
    search_handler: Any,
    prompt_config: Dict,
    search_method: str
) -> Dict[str, Any]:
    """Evaluate a single question."""
    try:
        if search_method == 'tag':
            inference = TagBasedInference(model, search_handler, prompt_config)
        elif search_method == 'function':
            inference = FunctionInference(model, search_handler, prompt_config)
        else:
            raise ValueError(f"Unknown search method: {search_method}")

        result = inference.run(question)
        return result
    except Exception as e:
        print(f"Error evaluating question: {e}")
        return {
            'answer': None,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="Run evaluation")
    parser.add_argument('--model', type=str, help='Model to evaluate')
    parser.add_argument('--method', type=str, choices=['tag', 'function'], help='Search method')
    parser.add_argument('--datasets', nargs='+', help='Datasets to evaluate')
    parser.add_argument('--config_dir', default='./evaluations/config', help='Config directory')
    parser.add_argument('--output_dir', default='./evaluations/results', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--checkpoint_every', type=int, default=None, help='Checkpoint frequency (overrides config)')

    args = parser.parse_args()

    print("Loading configurations...")
    configs = load_config(args.config_dir)

    model_name = args.model or configs['models']['active_model']
    datasets_to_eval = args.datasets or configs['datasets']['active_datasets']
    search_method = args.method or configs['search_engines']['search_engine']['search_method']

    checkpoint_every = args.checkpoint_every
    if checkpoint_every is None:
        checkpoint_every = configs['datasets'].get('evaluation', {}).get('checkpoint_every', 100)

    print(f"Checkpoint every: {checkpoint_every} examples")

    print(f"Model: {model_name}")
    print(f"Method: {search_method}")
    print(f"Datasets: {datasets_to_eval}")

    print("Initializing model...")
    model = initialize_model(model_name, configs)

    print("Initializing search...")
    search_handler = initialize_search(configs, search_method)

    prompt_config = get_prompt_config(configs, model_name, search_method)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"{model_name}_{search_method}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump({
            'model': model_name,
            'method': search_method,
            'datasets': datasets_to_eval,
            'timestamp': timestamp
        }, f, indent=2)

    all_results = {}

    for dataset_name in datasets_to_eval:
        print(f"\nEvaluating {dataset_name}...")

        dataset_config = configs['datasets']['datasets'][dataset_name]
        dataset = BenchmarkDataset(dataset_config)
        data = dataset.load()

        print(f"Loaded {len(data)} examples")

        results = []
        checkpoint_file = os.path.join(run_dir, f"{dataset_name}_checkpoint.jsonl")

        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                for line in f:
                    results.append(json.loads(line))
            print(f"Resumed from checkpoint: {len(results)} completed")

        for i in tqdm(range(len(results), len(data)), desc=f"Evaluating {dataset_name}"):
            item = data[i]

            result = evaluate_single(
                item['question'],
                model,
                search_handler,
                prompt_config,
                search_method
            )

            simplified_result = {
                'id': item['id'],
                'question': item['question'],
                'gold_answer': item['answers'][0] if item['answers'] else '',  # Use first answer as gold
                'prediction': result.get('answer', '')
            }

            if search_method == 'tag':
                simplified_result['response'] = result.get('response', '')
            elif search_method == 'function':
                simplified_result['messages'] = result.get('messages', [])

            result = simplified_result

            results.append(result)

            if (i + 1) % checkpoint_every == 0:
                with open(checkpoint_file, 'a') as f:
                    for r in results[-checkpoint_every:]:
                        f.write(json.dumps(r) + '\n')
                print(f"\nCheckpoint saved at {i + 1}")

        with open(checkpoint_file, 'w') as f:
            for r in results:
                f.write(json.dumps(r) + '\n')

        metrics = calculate_metrics(results, dataset_config['metrics'])

        dataset_results = {
            'dataset': dataset_name,
            'num_examples': len(results),
            'metrics': metrics,
            'results': results
        }

        with open(os.path.join(run_dir, f"{dataset_name}_results.json"), 'w') as f:
            json.dump(dataset_results, f, indent=2)

        all_results[dataset_name] = metrics

        print(f"\nMetrics for {dataset_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    summary = {
        'model': model_name,
        'method': search_method,
        'timestamp': timestamp,
        'results': all_results
    }

    with open(os.path.join(run_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nEvaluation complete! Results saved to {run_dir}")

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for dataset, metrics in all_results.items():
        print(f"\n{dataset}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()