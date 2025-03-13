from typing import List
import re
from langchain_core.output_parsers import StrOutputParser

def recursive_extract(text, pattern, default_answer):
    """
    Recursively extract the answer from the text using the pattern.

    Examples:
    - Input: "Assistant: The answer is 42."
        Output: "The answer is 42."
    - Input: "AI: The answer is 42."
        Output: "The answer is 42."
    - Input: "Assistant: The answer is 42. AI: The answer is 42."
        Output: "The answer is 42."
    """
    match = re.search(pattern, text, re.DOTALL)
    if match:
        assistant_text = match.group(1).strip()
        return recursive_extract(assistant_text, pattern, assistant_text)
    else:
        return default_answer

class Str_OutputParser(StrOutputParser):
    """
    Custom output parser for chat responses.
    """
    def parse(self, text: str) -> str:
        return self._extract_answer(text)
    
    def _extract_answer(self,
                       text_response: str,
                       patterns: List[str] = None,
                       default = "Sorry, I am not sure how to help with that."
                       ) -> str:
        """
        Extract the answer from the text response using the specified patterns."
        """
        input_text = text_response
        default_answer = default
        if patterns is None:
            patterns = [r'\nAssistant:(.*)', r'\nAI:(.*)']
        for pattern in patterns:
            output_text = recursive_extract(input_text, pattern, default_answer)
            if output_text != default_answer:
                input_text = output_text
                default_answer = output_text
        return output_text
