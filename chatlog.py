import os

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv('API_URL')

def init_streamlit():
    st.title("로그 분석기")

def show_chatlog():
    st.write("채팅 로그")
    url = f"{API_URL}/chatlog"
    
    resp = requests.get(url)
    resp = resp.json()

    # # 데이터를 테이블로 나타냄
    # st.table(resp)

    # JSON 응답을 DataFrame으로 변환
    df = pd.DataFrame(resp)

    # # 개행 문자 처리를 위해 문자열 컬럼을 HTML로 변환 (예시)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.replace('\n', '<br>', regex=False)

    # # Streamlit으로 DataFrame 렌더링
    # st.write(df.to_html(escape=False), unsafe_allow_html=True)

    # 열 순서 재정렬
    cols = ['User Input', 'RAG References', 'References Source', 'References Text Page', 'llama Response']
    df = df[cols]

    # 새로운 열 생성 및 포맷팅
    html_rows = []
    for idx, row in df.iterrows():
        html_row = f'<tr><td>{idx}</td><td>'
        html_row += f'<table><tr><td><b>User Input:</b></td><td>{row["User Input"]}</td></tr>'
        html_row += f'<tr><td><b>RAG References:</b></td><td>{row["RAG References"]}</td></tr>'
        html_row += f'<tr><td><b>References Source:</b></td><td>{row["References Source"]}</td></tr>'
        html_row += f'<tr><td><b>References Text Page:</b></td><td>{row["References Text Page"]}</td></tr>'
        html_row += f'<tr><td><b>GPT Response:</b></td><td>{row["llama Response"]}</td></tr></table></td></tr>'
        html_rows.append(html_row)

    html_table = '<table>' + ''.join(html_rows) + '</table>'

    # Streamlit으로 HTML 테이블 렌더링
    st.write(html_table, unsafe_allow_html=True)

if __name__ == "__main__":
    init_streamlit()
    show_chatlog()
