import os
import time

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv('API_URL')

def request_chat_api(user_message: str) -> str:
    url = f"{API_URL}/chat"

    resp = requests.post(
        url,
        json={
            "user_message": user_message,
        },
    )
    # 파이선 dict로 역직렬화
    resp = resp.json()
    resp = resp["answer"]
    print(resp)
    return resp

def init_streamlit():
    st.title("K-chat")

    # initial chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 채팅 메세지 보여주기
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

def chat_main():
    # chat_input을 message에 할당하되, 빈 문자열일때만(엔터만 눌렀을때) 조건문
    if message := st.chat_input(""):
        # 사용자 메세지를 chat history에 추가
        st.session_state.messages.append({"role": "user", "content": message})

        # 유저 메세지 보여주기
        with st.chat_message("user"):
            st.markdown(message)

        # api대답 받아서 보여주기 > chat api 호출
        assistant_response = request_chat_api(message)

        # 챗봇 메세지를 chat history에 추가
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for lines in assistant_response.split("\n"):
                for chunkl in lines.split():
                    full_response += chunkl + " "
                    time.sleep(0.05)
                    # add a blinking cursor
                    message_placeholder.markdown(full_response)
                full_response += "\n"
            message_placeholder.markdown(full_response)

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )

if __name__ == "__main__":
    init_streamlit()
    chat_main()
