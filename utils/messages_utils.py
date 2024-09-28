from typing import List
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from enum import Enum


class MessageType(Enum):
    HUMAN = "Human"
    AI = "Ai"


def process_chat_history_from_request(req_chat_history: list) -> List[BaseMessage]:
    chat_history = []
    for item in req_chat_history:
        type = item["type"]
        content = item["message"]

        if MessageType(type) == MessageType.HUMAN:
            chat_history.append(HumanMessage(content=content))
        else:
            chat_history.append(AIMessage(content=content))
    return chat_history
