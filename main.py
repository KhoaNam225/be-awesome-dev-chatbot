import json

from utils.messages_utils import process_chat_history_from_request
from agents.question_answer_agent import create_qa_agent


def lambda_handler(events, context):
    if events["body"] is None:
        return {"statusCode": "400", "body": "Request must contain message"}

    request_body = json.loads(events["body"])

    chat_history = (
        request_body["chatHistory"] if request_body["chatHistory"] is not None else []
    )

    new_message = request_body["newMessage"]

    if new_message == "" or new_message is None:
        return {"statusCode": "400", "body": "Request must contain message"}

    chat_history = process_chat_history_from_request(chat_history)

    qa_agent = create_qa_agent()
    response = qa_agent.invoke({"input": new_message, "chat_history": chat_history})
    ai_answer = response["answer"]

    return {"body": f"{ai_answer}"}
