import ast
import os

import numpy as np
import openai
import pandas as pd
from dotenv import load_dotenv
from numpy import dot
from numpy.linalg import norm
from openai import OpenAI

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI()


# openai embedding함수
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def cos_sim(A, B):
    return dot(A, B)/(norm(A)*norm(B))

def query_csv(query: str, use_retriever: bool = False):
    # 입력된 문장 임베딩
    query_embedding = get_embedding(
        query,
        model="text-embedding-3-small"
    )
    if use_retriever:
        # csv파일을 읽어서 임베딩값과 가장 가까운 3개 문장을 반환
        df = pd.read_csv('./database/embedding.csv')

        # 문자열로 저장된 embedding을 실제 숫자 배열로 변환
        df['embedding'] = df['embedding'].apply(ast.literal_eval)

        df['similarity'] = df.embedding.apply(lambda x: cos_sim(np.array(x), np.array(query_embedding)))
        top_docs = df.sort_values('similarity', ascending=False).head(3)
    else:
        # TODO: 다른 경우 처리
        top_docs = []

    return top_docs
