# https://github.com/aurelio-labs/semantic-router
# https://python.langchain.com/docs/integrations/chat/google_generative_ai/
# https://python.langchain.com/docs/how_to/chat_model_caching/
# https://python.langchain.com/docs/integrations/memory/redis_chat_message_history/
# https://python.langchain.com/docs/integrations/memory/
# https://discuss.streamlit.io/t/how-to-launch-streamlit-app-from-google-colab-notebook/42399

import uuid

import requests
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_community.cache import SQLiteCache
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.globals import set_llm_cache
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

# https://python.langchain.com/docs/integrations/chat/huggingface/
from langchain_google_genai import ChatGoogleGenerativeAI
from semantic_router import Route
from semantic_router.encoders import OpenAIEncoder
from semantic_router.routers import SemanticRouter

load_dotenv()


interviewee_route = Route(
    name="interviewee",
    utterances=[
        "Who is Steve?",
        "Software developer",
        "Which skill does Steve have?",
        "Which framework does Steve have?",
        "Which exp does Steve have?",
        "Which experience does Steve have?",
        "How many years of experience does Steve have?",
        "Which experience does Steve have?",
        "Descripbe Steve",
        "What is Steve's skill?",
        "Describe skills of Steve",
        "What is Steve's framework?",
    ],
)
chitchat_route = Route(
    name="chitchat",
    utterances=[
        "What's the weather like today?",
        "How hot is it outside?",
        "Will it rain tomorrow?",
        "What is the current temperature?",
        "Can you tell me what the current weather conditions are?",
        "Will it be sunny this weekend?",
        "What was the temperature yesterday?",
        "How cold will it be tonight?",
        "Who was the first president of the United States?",
        "In what year did World War II end?",
        "Can you tell me about the history of the internet?",
        "In what year was the Eiffel Tower built?",
        "Who invented the telephone?",
        "What is your name?",
        "Do you have a name?",
        "What should I call you?",
        "Who created you?",
        "How old are you?",
        "Can you tell me a fun fact?",
        "Do you know any interesting trivia?",
        "What is your favorite color?" "What is your favorite movie?",
        "Do you have any hobbies?",
        "What is the meaning of life?",
        "Can you tell me a joke?",
        "What is the capital of France?",
        "What is the population of the world?",
        "How many continents are there?",
        "Who wrote 'To Kill a Mockingbird'?",
        "Can you give me a quote by Albert Einstein?",
    ],
)
routes = [interviewee_route, chitchat_route]
encoder = OpenAIEncoder()
route_layer = SemanticRouter(encoder=encoder, routes=routes, auto_sync="local")

url = "https://stevedev.pythonanywhere.com/"  # Replace with the desired URL
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")
steve_text_content = soup.get_text()
steve_text_content = soup.get_text()
steve_clean_text = steve_text_content.replace("\n", " ").strip()

# Optional: remove extra spaces
steve_clean_text = " ".join(steve_clean_text.split())


# https://python.langchain.com/docs/integrations/chat/google_generative_ai/
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


# https://alejandro-ao.com/how-to-use-streaming-in-langchain-and-streamlit/#what-is-lcel
# app config
st.set_page_config(page_title="Steve's Assitant", page_icon="🤖")
st.title("Steve's Assitant")
st.logo("avatar.jpg")
avatars = {
    "assistant": "avatar.jpg",
    # "user": "https://ui-avatars.com/api/?rounded=true&name=user"
    "user": None,
}


set_llm_cache(SQLiteCache(database_path=".langchain.db"))


def get_session_history(session_id: str):
    return InMemoryChatMessageHistory()


chats_by_session_id = {}


def get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
    chat_history = chats_by_session_id.get(session_id)
    if chat_history is None:
        chat_history = InMemoryChatMessageHistory()
        chats_by_session_id[session_id] = chat_history
    return chat_history


msgs = StreamlitChatMessageHistory(key="special_app_key")


def get_response(user_query, chat_history, session_id):
    print("===== session_id ", session_id)
    # The first time, it is not yet in cache, so it should take longer
    output = route_layer(user_query)
    print("output", output)
    # chat_model = ChatHuggingFace(llm=llm)

    if output.name == "interviewee":
        # set_llm_cache(InMemoryCache())
        set_llm_cache(SQLiteCache(database_path=".langchain.db"))
        prompt = ChatPromptTemplate.from_template(
            "You are assisstant, answer {user_question} base on infomation of developers: {data} and chat History: {chat_history}"
        )
        chain = prompt | chat_model | StrOutputParser()
        # chain.invoke({"question": input_text,"data": results})
        chain_with_history = RunnableWithMessageHistory(
            chain,
            get_session_history=get_chat_history,
            input_messages_key="user_question",  # which input is treated as user message
            history_messages_key="chat_history",  # where past messages will be injected
        )
        return chain_with_history.stream(
            {
                "chat_history": chat_history,
                "user_question": user_query,
                "data": steve_clean_text,
            },
            config={"configurable": {"session_id": session_id}},
        )
    else:
        # set_llm_cache(InMemoryCache())
        set_llm_cache(SQLiteCache(database_path=".langchain.db"))
        chat_model = llm
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an AI chatbot having a conversation with a human."),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{user_question}"),
            ]
        )
        output_parser = StrOutputParser()
        chain = prompt | chat_model | output_parser
        print("=== msgs ", msgs)
        chain_with_history = RunnableWithMessageHistory(
            chain,
            lambda session_id: msgs,  # Always return the instance created earlier
            input_messages_key="user_question",
            history_messages_key="history",
        )
        return chain_with_history.stream(
            {
                "user_question": user_query,
            },
            config={"configurable": {"session_id": session_id}},
        )



if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    # st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("User"):
        st.markdown(user_query)

    with st.chat_message("Assistant", avatar=avatars["assistant"]):
        response = st.write_stream(
            get_response(
                user_query, "", st.session_state.session_id
            )
        )
        # st.write(response)

    # st.session_state.chat_history.append(AIMessage(content=response))
