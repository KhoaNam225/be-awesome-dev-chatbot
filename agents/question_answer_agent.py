from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from utils.llm_utils import init_chat_model
from agents.history_aware_retriever_agent import create_retriever


def create_qa_agent():
    llm_for_qa = init_chat_model()
    qa_system_prompt = """
  You are an assistant for question-answering tasks related to content of website called beAwesome.dev.
  This website contains content about programming knowledge in general.
  Use the following pieces of retrieved context to answer the question.
  If you don't know the answer, just say that you don't know.
  
  Answer the question as descriptive and clear as possible. Give some code examples if possible.
  If there is no context provided for you, that means the question is not asking about something relevant to the website content.
  In the case, reply politely and ask the user to check the question again.

  If there is no context provided to you, reply to the user that the website doesn't have information related to the question asked. However, if the user doesn't ask any question and just greet you, reply appropriately.

  Here is the context from the website, answer the question based on the information in the context:

  {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm_for_qa, qa_prompt)
    retriever = create_retriever()
    qa_agent = create_retrieval_chain(retriever, question_answer_chain)

    return qa_agent
