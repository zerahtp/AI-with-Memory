from http.client import responses
from itertools import chain

from dotenv import load_dotenv  # .env dosyasını yükleyerek API anahtarlarını kullanabilir hale getiriyoruz
from langchain_openai import ChatOpenAI  # OpenAI'nin dil modelini kullanmak için LangChain wrapper'ı
from langchain_core.messages import HumanMessage, AIMessage  # Kullanıcı ve AI mesajlarını temsil eden sınıflar
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)  # Geçmiş mesajları tutan hafıza mekanizması
from langchain_core.runnables.history import RunnableWithMessageHistory  # Mesaj geçmişi ile çalışabilen LangChain bileşeni
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # Prompt şablonu oluşturmak için
from sqlalchemy.testing.plugin.plugin_base import config  # SQLAlchemy test eklentisi (Gereksiz gibi görünüyor)
from sqlalchemy.testing.suite.test_reflection import users  # SQLAlchemy test modülü (Gereksiz olabilir)

# Çevresel değişkenleri yükleyerek API anahtarlarını kullanabilir hale getiriyoruz
load_dotenv()

# OpenAI modelini başlatıyoruz. GPT-3.5 Turbo modelini kullanıyoruz.
model = ChatOpenAI(model="gpt-3.5-turbo")

# Kullanıcı oturum geçmişini saklamak için bir sözlük oluşturuyoruz
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Belirtilen oturum kimliği için mesaj geçmişini döndürür.
    Eğer daha önce oluşturulmadıysa, yeni bir oturum başlatır.
    """
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# ChatBot'un genel yönlendirmesi için sistem mesajı içeren bir prompt şablonu oluşturuyoruz
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a helpful assistant."),  # Modelin genel rolü: yardımcı bir asistan
        MessagesPlaceholder(variable_name="message"),  # Kullanıcının göndereceği mesajları tutacak değişken
    ]
)

# Prompt şablonu ve model arasında bir zincir oluşturuyoruz
chain = prompt | model

# Kullanıcıların oturum kimliğini belirten bir konfigürasyon oluşturuyoruz
# Bu, AI'nin belirli bir oturuma ait mesaj geçmişini hatırlamasını sağlar
config = {"configurable": {"session_id": "firstChat"}}

# Belirli bir oturumun mesaj geçmişini kullanarak modeli çalıştıran bir nesne oluşturuyoruz
with_message_history = RunnableWithMessageHistory(chain, get_session_history)

if __name__ == "__main__":
    # Terminal üzerinden kullanıcı girişi alarak sohbeti başlatıyoruz
    while True:
        user_input = input("> ")  # Kullanıcıdan giriş al
        response = with_message_history.invoke(
            [HumanMessage(content=user_input)],  # Kullanıcının mesajını modele gönder
            config=config,  # Oturum bilgisi ile birlikte modeli çalıştır
        )
        print(response.content)  # Modelin cevabını ekrana yazdır

        # Eğer cevabı parça parça (stream olarak) almak istersek şu alternatifi kullanabiliriz:
        # while True:
        #     user_input = input("> ")
        #     for r in with_message_history.stream(
        #             [HumanMessage(content=user_input)],
        #             config=config,
        #     ):
        #         print(r.content, end=" ")
