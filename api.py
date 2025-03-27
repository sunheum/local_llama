import os
from typing import Dict

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from chain import read_prompt_template, create_prompt

from fastapi.middleware.cors import CORSMiddleware

# 1. 시스템 템플릿 설정
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
SYSTEM_MESSAGE_TEMPLATE = os.path.join(CUR_DIR, "prompt_templates", "system_message.txt")
system_message = read_prompt_template(SYSTEM_MESSAGE_TEMPLATE)

# 2. 모델 생성
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key)

# model_path = "/data/workspace/Llama-3-Alpha-Ko-8B-Instruct"

# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     torch_dtype="auto",
#     device_map="auto",
# )
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "{system_message}"),
        MessagesPlaceholder(variable_name="history"), # 대화 기록을 변수로 사용, history 가 MessageHistory 의 key 가 됨
        ("human", "{user_message}"),
    ]
)
runnable = prompt | model  # 프롬프트와 모델을 연결하여 runnable 객체 생성

# 3. 세션 기록을 저장할 딕셔너리 생성한 모델생성
store = {}  # 빈 딕셔너리를 초기화합니다.

def get_session_history(user_id: str, conversation_id: str) -> BaseChatMessageHistory:
    # 주어진 user_id와 conversation_id에 해당하는 세션 기록을 반환합니다.
    if (user_id, conversation_id) not in store:
        # 해당 키가 store에 없으면 새로운 ChatMessageHistory를 생성하여 저장합니다.
        store[(user_id, conversation_id)] = ChatMessageHistory()
    return store[(user_id, conversation_id)]

with_message_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="user_message",
    history_messages_key="history",
    history_factory_config=[  # 기존의 "session_id" 설정을 대체하게 됩니다.
        ConfigurableFieldSpec(
            id="user_id",  # get_session_history 함수의 첫 번째 인자로 사용됩니다.
            annotation=str,
            name="User ID",
            description="사용자의 고유 식별자입니다.",
            default="",
            is_shared=True,
        ),
        ConfigurableFieldSpec(
            id="conversation_id",  # get_session_history 함수의 두 번째 인자로 사용됩니다.
            annotation=str,
            name="Conversation ID",
            description="대화의 고유 식별자입니다.",
            default="",
            is_shared=True,
        ),
    ],
)

# 4. 대화 실행
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

class UserRequest(BaseModel):
    user_message: str

def create_chat_log_csv(user_input: str, use_rag: bool, rag_references: str, references_source: str, references_text_page: str, llama_response: str):
    # CSV 파일 경로를 설정합니다.
    csv_file_path = os.path.join('./chat_log', 'chat_logs.csv')
    # 새로운 로그 데이터를 DataFrame으로 생성합니다.
    new_log = pd.DataFrame({
        'User Input': [user_input],
        'Use RAG': [use_rag],
        'RAG References': [rag_references],
        'References Source': [references_source],
        'References Text Page': [references_text_page],
        'llama Response': [llama_response]
    })

    # 파일이 존재하지 않거나 비어 있는 경우, 헤더를 포함하여 새로운 데이터를 저장합니다.
    # 그렇지 않으면, 헤더 없이 데이터만 추가합니다.
    if not os.path.isfile(csv_file_path) or os.stat(csv_file_path).st_size == 0:
        # 파일이 존재하지 않거나 비어 있으면, 새 파일을 생성하고 헤더를 포함하여 데이터를 저장합니다.
        new_log.to_csv(csv_file_path, mode='w', index=False, encoding='utf-8-sig', sep=';', header=True)
    else:
        # 파일이 이미 존재하고 비어 있지 않으면, 기존 데이터에 새 로그를 추가합니다(헤더 없음).
        new_log.to_csv(csv_file_path, mode='a', index=False, encoding='utf-8-sig', sep=';', header=False)


@app.post("/chat")
def generate_answer(req: UserRequest) -> Dict[str, str]:
    user_input = req["user_message"]

    # RAG를 한 시스템 메세지 생성
    use_rag = True
    system_message, references_source, references_text, references_text_page = create_prompt(user_input)

    # 사용자 입력을 받아서 대화를 실행합니다.
    answer = with_message_history.invoke(
        {
            "user_message": user_input,
            "system_message": system_message, 
        },
        config={"configurable": {"user_id": "test1", "conversation_id": "test_conv1"}},
    )

    # 대화내용을 저장합니다
    create_chat_log_csv(
        user_input,
        use_rag,
        # [message for message in messages if message['role'] == 'system'],
        references_text,
        references_source,
        references_text_page,
        answer.content)

    return {
        "answer": answer.content,
    }


# 상담내용
@app.get("/chatlog")
async def read_csv_data():
    csv_file_path = './chat_log/chat_logs.csv'
    data = pd.read_csv(csv_file_path, sep=';')
    return data.to_dict(orient='records')

