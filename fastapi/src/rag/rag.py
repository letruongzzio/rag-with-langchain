import re # Used for processing the output of the RAG model and extracting the answer text
from langchain import hub # Used for loading the RAG prompt
from langchain_core.runnables import RunnablePassthrough # Used for passing the documents to the RAG model
from langchain_core.output_parsers import StrOutputParser # Used for parsing the output of the RAG model


class Str_OutputParser(StrOutputParser):
    """
    This class is used to parse the output of the RAG model.

    Explanation: The pattern `r"Answer:\s*(.*)"` is a regular expression used to search and extract a portion of a text string. 
    Let's break down each part of this pattern:
    
    1. `r""`: This is a raw string literal in Python. When using `r""`, special characters in the string will not be escaped. 
       This is useful when working with regular expressions because it helps avoid having to escape special characters multiple times.
       For instance:
           >>> print(r"\n")  # Output: \n (raw string contains the backslash character)
           >>> print("\n")   # Output: (newline character)
    
    2. `Answer:`: This is a fixed string that the regular expression will search for. It requires the text string to contain the word "Answer:".
    
    3. `\s*`:
       - `\s` is a special character in regular expressions, representing any whitespace character (including spaces, tabs, newlines, etc.).
       - `*` is a repetition operator, meaning "0 or more times".
       So, `\s*` means "0 or more whitespace characters".
    
    4. `(.*)`:
       - `.` is a special character in regular expressions, representing any character except newline.
       - The pair of parentheses `()` is used to create a capture group.
       In this case, `(.*)` will capture and store all characters after "Answer:" and any whitespace, up to the end of the line or string.
    
    In short, the pattern `r"Answer:\s*(.*)"` will search for a string that starts with "Answer:", followed by any number of whitespace characters, 
    and then any sequence of characters. The part of the string after "Answer:" and the space will be stored in the first catch group (group 1) 
    and can be accessed using `match.group(1)`.
    
    For example:
    - For the string "Answer: Hello World", `group(1)` will return "Hello World".
    - For the string "Answer: This is a test", `group(1)` will return "This is a test".
    """
    def parse(self, text: str) -> str:
        return self._extract_answer(text)
    
    def _extract_answer(self,
                       text_response: str,
                       pattern: str = r"Answer:\s*(.*)"
                       ) -> str:
        match = re.search(pattern, text_response, re.DOTALL)
        # `re.DOTALL` is used to match any character including newline
        if match:
            answer_text = match.group(1).strip()
            return answer_text
        else:
            return text_response

class RAG:
    """
    This class is used to create a RAG chain.
    """
    def __init__(self, llm) -> None:
        self.llm = llm
        self.prompt = hub.pull("rlm/rag-prompt")
        self.str_parser = Str_OutputParser()
    
    def get_chain(self, retriever):
        """
        This method is used to create a RAG chain.

        The RAG chain consists of the following components:
        - Input data: The context (documents retrieved by the retriever) and the question
        - RAG prompt: The RAG prompt used to generate the answer
        - LLM: The language model used to generate the answer
        - String output parser: The output parser used to extract the answer text from the model output

        The RAG chain takes the input data, passes it through the RAG prompt, generates an answer using the language model,
        and then extracts the answer text from the model output.

        Example:
        - Input data: {"context": [doc1, doc2, doc3], "question": "What is the capital of France?"}
        - Process:
            1. Pass the context
            2. Generate the RAG prompt: "Generate an answer to the question: What is the capital of France? using the following documents: doc1, doc2, doc3"
            3. Pass the prompt and context to the language model
            4. Extract the answer text from the model output: "Answer: Paris"
        - Output: "Paris"
        """
        input_data = {
            "context": retriever | self._format_docs, 
            "question": RunnablePassthrough()
        }
        rag_chain = (
            input_data
            | self.prompt
            | self.llm
            | self.str_parser
        )
        return rag_chain

    def _format_docs(self, docs):
        """
        This method is used to format the documents in the required format.

        Example:
        - Input: [doc1, doc2, doc3]
        - Output: "doc1\n\ndoc2\n\ndoc3"
        """
        return "\n\n".join(doc.page_content for doc in docs)
