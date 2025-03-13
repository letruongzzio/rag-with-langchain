"""
This module defines a factory function to manage chat history by session.

The factory function `create_session_factory` creates a function that retrieves chat history from a JSON file by session_id. The function checks the format of the session_id and limits the number of stored messages to a specified maximum history length.

The `create_session_factory` function takes two arguments:
- `base_dir`: The base directory where the chat history files are stored.
- `max_history_length`: The maximum number of messages to store in the chat history.

The factory function returns a function `get_chat_history` that takes a session_id as input and returns a `FileChatMessageHistory` object that manages the chat history for that session.
"""

import os
import re
from typing import Callable, Union
from fastapi import HTTPException
from langchain_core.chat_history import BaseChatMessageHistory # BaseChatMessageHistory is an abstract class that defines the interface for chat history management.
from langchain.memory import FileChatMessageHistory # FileChatMessageHistory is a class that implements chat history management using a file-based storage system.

def _is_valid_identifier(value: str) -> bool:
    """
    Check if the session_id is valid (only contains letters, numbers, hyphens, and underscores):
    - Letters: a-z, A-Z
    - Numbers: 0-9
    - Hyphens: -
    - Underscores: _

    Examples:
    - "chat_session_123" is valid.
    - "session-001" is valid.
    - "session@123" is invalid.
    - "session 123" is invalid.
    """
    valid_characters = re.compile(r"^[a-zA-Z0-9-_]+$")
    return bool(valid_characters.match(value))

def create_session_factory(base_dir: Union[str, str], max_history_length: int) -> Callable[[str], BaseChatMessageHistory]:
    """Create a factory to manage chat history by session."""

    # Ensure the directory exists, create it if it doesn't
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    def get_chat_history(session_id: str) -> FileChatMessageHistory:
        """Retrieve chat history from a JSON file by session_id."""
        
        # Check the format of session_id
        if not _is_valid_identifier(session_id):
            raise HTTPException(
                status_code=400,
                detail=f"Session ID `{session_id}` is invalid. "
                       "It can only contain letters, numbers, hyphens (-), and underscores (_)."
            )

        # Determine the file path to save chat history
        file_path = os.path.join(base_dir, f"{session_id}.json")

        # Create an object to manage chat history
        chat_hist = FileChatMessageHistory(file_path)
        messages = chat_hist.messages # Get the list of messages stored in the chat history

        # Limit the number of stored messages
        if len(messages) > max_history_length:
            chat_hist.clear()  # Clear old history
            for message in messages[-max_history_length:]:  # Keep only the last max_history_length messages
                chat_hist.add_message(message)

        print("Number of messages in history: ", len(chat_hist.messages))
        return chat_hist

    return get_chat_history
