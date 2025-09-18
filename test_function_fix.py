#!/usr/bin/env python3
"""Test script to verify function calling fix."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'evaluations'))

from evaluations.run_evaluation import load_config, initialize_model, initialize_search, get_prompt_config
from evaluations.src.inference.function_inference import FunctionInference

def test_function_calling():
    """Test function calling with a sample question."""

    # Load configurations
    config_dir = './evaluations/config'
    configs = load_config(config_dir)

    # Use GPT-4 (closed source) with function method
    model_name = 'gpt-4'
    search_method = 'function'

    # Initialize components
    model = initialize_model(model_name, configs)
    search_handler = initialize_search(configs, search_method)
    prompt_config = get_prompt_config(configs, model_name, search_method)

    # Create inference instance
    inference = FunctionInference(model, search_handler, prompt_config)

    # Test question that requires multiple searches
    test_question = "Who was president of the United States in the year that Citibank was founded?"

    print(f"Testing question: {test_question}")
    print("=" * 50)

    # Run inference
    result = inference.run(test_question)

    # Check results
    if result.get('answer'):
        print(f"✓ Answer found: {result['answer']}")
    else:
        print(f"✗ No answer found")

    # Check number of messages (should be more than 4 if multiple tool calls were made)
    num_messages = len(result.get('messages', []))
    print(f"Number of messages in conversation: {num_messages}")

    # Print message types to verify flow
    print("\nMessage flow:")
    for i, msg in enumerate(result.get('messages', [])):
        role = msg.get('role', 'unknown')
        has_tool_calls = 'tool_calls' in msg
        has_content = bool(msg.get('content'))

        if role == 'assistant' and has_tool_calls:
            print(f"  {i+1}. {role} (with tool_calls)")
        elif role == 'tool':
            print(f"  {i+1}. {role} response")
        else:
            print(f"  {i+1}. {role}")

    return result

if __name__ == "__main__":
    try:
        result = test_function_calling()

        # Print full messages for debugging if needed
        if '--verbose' in sys.argv:
            print("\n" + "=" * 50)
            print("Full message history:")
            import json
            print(json.dumps(result.get('messages', []), indent=2))

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()