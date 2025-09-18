"""Function-based inference implementation."""

from typing import Dict, Any, List
from ..search.function_search import FunctionSearchHandler


class FunctionInference:
    """Handle function-based inference with multiple tool support."""

    def __init__(self, model, search_handler: FunctionSearchHandler, prompt_config: Dict):
        """
        Initialize function inference.

        Args:
            model: Language model instance
            search_handler: Function search handler
            prompt_config: Prompt configuration
        """
        self.model = model
        self.search_handler = search_handler
        self.prompt_config = prompt_config
        self.max_iterations = 10

    def run(self, question: str) -> Dict[str, Any]:
        """
        Run inference with function tools.

        Args:
            question: Input question

        Returns:
            Dictionary containing answer, messages, tool calls, and iterations
        """
        # Initialize clean message history
        messages = []
        if self.prompt_config.get('system'):
            messages.append({
                "role": "system",
                "content": self.prompt_config['system']
            })
        messages.append({
            "role": "user",
            "content": self.prompt_config['user'].format(question=question)
        })

        tools = self.search_handler.get_tool_schemas()

        all_function_calls = []
        iterations = 0
        final_answer = None

        try:
            while iterations < self.max_iterations:
                iterations += 1

                # Generate response with functions
                response = self.model.generate_with_functions(messages, tools)

                # Parse function calls from response
                function_calls = self.search_handler.parse_tool_calls(response) # check if <tool_call>...</tool_call> exists in the response

                if function_calls:
                    messages.append({
                        "role": "assistant",
                        "content": response.get('content', ''),
                        "tool_calls": response.get('tool_calls', [])
                    })

                    # Execute each function call
                    for call in function_calls:
                        function_name = call['name']
                        arguments = call['arguments']

                        result = self.search_handler.call_function(function_name, arguments)

                        tool_response = self.search_handler.format_tool_response(
                            call['id'],
                            result
                        )
                        messages.append(tool_response)

                        # Track function call for summary (with shortened result)
                        all_function_calls.append({
                            'function': function_name,
                            'arguments': arguments,
                            'result': result[:200] if len(result) > 200 else result  # Store preview only
                        })

                else:
                    # No function calls, check for final answer
                    if response.get('content'):
                        # Add final assistant message
                        messages.append({
                            "role": "assistant",
                            "content": response['content']
                        })

                        # Try to extract answer
                        final_answer = self.search_handler.extract_final_answer(response['content'])

                        # If we have an answer, return results
                        if final_answer:
                            return self._format_results(
                                answer=final_answer,
                                messages=messages,
                                function_calls=all_function_calls,
                                iterations=iterations
                            )

                        # If no answer found but content exists, continue if we haven't reached max iterations
                        if iterations >= self.max_iterations:
                            # Use the last content as answer if no explicit answer tags
                            final_answer = response['content']
                            break
                    else:
                        # No content and no function calls - something went wrong
                        break

            return self._format_results(
                answer=final_answer,
                messages=messages,
                function_calls=all_function_calls,
                iterations=iterations
            )

        except Exception as e:
            return {
                'answer': None,
                'error': str(e),
                'messages': messages,
                'tool_calls': all_function_calls,  
                'iterations': iterations
            }

    def _format_results(self, answer: Any, messages: List[Dict], function_calls: List[Dict], iterations: int) -> Dict[str, Any]:
        """
        Format the results for output.

        Args:
            answer: Extracted answer
            messages: Complete message history
            function_calls: Summary of all function calls
            iterations: Number of iterations

        Returns:
            Formatted results dictionary
        """
        search_queries = []
        for fc in function_calls:
            if 'search' in fc.get('function', ''):
                query = fc.get('arguments', {}).get('query', '')
                if query:
                    search_queries.append(query)

        return {
            'answer': answer,
            'messages': messages,  # Clean message history
            'tool_calls': function_calls,  # Summary of function calls (backward compatibility)
            'search_queries': search_queries,  # For backward compatibility with metrics
            'iterations': iterations
        }