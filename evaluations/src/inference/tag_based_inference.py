"""Tag-based inference implementation."""

from typing import Dict, Any
from ..search.tag_search import TagBasedSearch


class TagBasedInference:
    """Handle tag-based search inference."""

    def __init__(self, model, search_handler: TagBasedSearch, prompt_config: Dict):
        self.model = model
        self.search_handler = search_handler
        self.prompt_config = prompt_config
        self.max_iterations = 10

    def run(self, question: str) -> Dict[str, Any]:
        """Run inference with tag-based search."""
        prompt = self.prompt_config['user'].format(question=question)

        iterations = 0
        full_response = ""

        while iterations < self.max_iterations:
            iterations += 1

            stop_sequences = [
                "</search>", " </search>",
                "</answer>", " </answer>"
            ]

            response = self.model.generate_with_tags(
                prompt,
                stop_sequences=stop_sequences,
                max_tokens=512
            )

            full_response += response
            prompt += response

            _, reason = self.search_handler.should_continue(full_response)

            if reason == "answer_found":
                answer = self.search_handler.extract_answer(full_response)
                return {
                    'answer': answer,
                    'response': full_response  
                }

            elif reason == "search_needed":
                query = self.search_handler.extract_search_query(response)
                if query:
                    results = self.search_handler.search_engine.search(query)
                    # results = "test_placeholder"
                    search_text = self.search_handler.format_search_results(results)
                    prompt += search_text
                    full_response += search_text

        return {
            'answer': None,
            'response': full_response  
        }