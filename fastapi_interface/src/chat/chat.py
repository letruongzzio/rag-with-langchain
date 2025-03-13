from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# `MessagesPlaceholder` is a placeholder for messages in a chat prompt template.
from langchain_core.runnables.history import RunnableWithMessageHistory # `RunnableWithMessageHistory` is a runnable that stores chat history.
from fastapi_interface.src.chat.history import create_session_factory
from fastapi_interface.src.chat.output_parser import Str_OutputParser

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer all questions to the best of your ability."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{human_input}"),
    ]
)

class InputChat(BaseModel):
    """
    Input data for the chat system.
    """
    human_input: str = Field(
        ...,
        description="The human input to the chat system.",
        extra={"widget": {"type": "chat", "input": "human_input"}},
    )

def build_chat_chain(llm, history_folder, max_history_length):
    """
    Build a chat chain with a history of chat messages."
    
    Args:
        llm: The language model used for chat.
        history_folder: The folder where chat history files are stored.
        max_history_length: The maximum number of messages to store in the chat history."
    
    Returns:
        A chat chain with a history of chat messages.
    """
    chain = chat_prompt | llm | Str_OutputParser()
    # Use `RunnableWithMessageHistory` to store chat history. It is a necessary step to enable chat history functionality and helps chatbots remember past conversations.
    chain_with_history = RunnableWithMessageHistory(
        runnable=chain,
        get_session_history=create_session_factory(
            base_dir=history_folder,
            max_history_length=max_history_length
        ),
        input_messages_key="human_input",
        history_messages_key="chat_history",
    )
    return chain_with_history.with_types(input_type=InputChat)
