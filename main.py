import json
import logging
from utils.messages_utils import process_chat_history_from_request
from agents.question_answer_agent import create_qa_agent


def lambda_handler(events, context):
    allowed_origins = ["http://localhost:3000", "https://djcogs8kyls7c.cloudfront.net"]
    origin = events.get("headers", {}).get("origin", "")

    if origin in allowed_origins:
        cors_headers = {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Credentials": "true",
        }
    else:
        cors_headers = {
            "Access-Control-Allow-Origin": "null",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Credentials": "true",
        }

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

    return {"body": f"{ai_answer}", "headers": cors_headers}
