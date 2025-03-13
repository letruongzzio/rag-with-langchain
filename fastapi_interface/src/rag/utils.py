import re

def extract_answer(text_response: str,
                   pattern: str = r"Answer:\s*(.*)"
                   ) -> str:
    """
    Extracts the answer from the text response.

    Args:
        text_response (str): The text response from the RAG model.
        pattern (str, optional): The regex pattern to extract the answer. Defaults to r"Answer:\s*(.*)".
    
    Returns:
        str: The extracted answer.
    """
    match = re.search(pattern, text_response)
    if match:
        answer_text = match.group(1).strip()
        return answer_text
    else:
        return "Answer not found."
