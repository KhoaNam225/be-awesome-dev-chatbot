from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from utils.llm_utils import init_chat_model, init_compression_retriever


def create_retriever():
    llm_for_retriever = init_chat_model()
    retriever = init_compression_retriever()
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
  which might reference context in the chat history, formulate a standalone question \
  which can be understood without the chat history. Do NOT answer the question, \
  just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm_for_retriever, retriever, contextualize_q_prompt
    )

    return history_aware_retriever
