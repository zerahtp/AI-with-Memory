from http.client import responses
from itertools import chain

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from sqlalchemy.testing.plugin.plugin_base import config
from sqlalchemy.testing.suite.test_reflection import users

load_dotenv()

model = ChatOpenAI(model ="gpt-3.5-turbo")

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","you are a helpful assistant."),
        MessagesPlaceholder(variable_name="message"),
    ]
)

chain = prompt | model
#model bu kısımda id ile hangi historyde olduğunu tutarak hafızasında tutuyor
config = {"configurable": {"session_id": "firstChat"}}
#hangi chaini çalıştırmak istediğimizi sessions historysini nerden alacağı fonksiyon
with_message_history = RunnableWithMessageHistory(chain,get_session_history)

if __name__ == "__main__":
    while True:
        user_input = input(">")
        response = with_message_history.invoke(
            #manuel messaj almayarak chatbot tarzı oldu
            [HumanMessage(content=user_input)],
            config=config,
        )
        print(response.content)

        #stream olarak cevabı almak istiyorsak
        #while True:
        #    user_input = input(">")
        #    for r in with_message_history.stream(
        #            [HumanMessage(content=user_input)],
        #            config=config,
        #    ):
        #        print(r.content,end=" ")
