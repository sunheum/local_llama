import ast
import os

import numpy as np
import openai
import pandas as pd
import faiss
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

def create_and_save_faiss_index(embeddings, index_path):
    # 첫 번째 임베딩을 기반으로 인덱스의 차원을 설정
    d = len(embeddings[0])
    index = faiss.IndexFlatL2(d)

    # 임베딩을 순회하면서 인덱스에 추가
    for embedding in embeddings:
        # 각 임베딩을 넘파이 배열로 변환하고 차원을 (1, d)로 재조정
        np_embedding = np.array(embedding, dtype='float32').reshape(1, -1)
        index.add(np_embedding)

    # 인덱스를 파일로 저장
    faiss.write_index(index, index_path)
    print(f"FAISS 인덱스가 {index_path}에 성공적으로 저장되었습니다.")


# 디렉토리 설정
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_DIR = os.path.join(CUR_DIR, 'faiss_index')
FAISS_INDEX_NAME = "security_guide.index"

# FAISS 인덱스를 저장할 디렉토리가 없다면 생성
if not os.path.exists(FAISS_INDEX_DIR):
    os.makedirs(FAISS_INDEX_DIR)

# 인덱스 파일 경로
index_path = os.path.join(FAISS_INDEX_DIR, FAISS_INDEX_NAME)

# embedding.csv 파일 경로
folder_path = './database'
embedding_file = 'embedding.csv'
embedding_file_path = os.path.join(folder_path, embedding_file)
# embedding.csv가 존재하면, 데이터프레임 df로 로드
if os.path.exists(embedding_file_path):
    print(f"{embedding_file} is exist")
    df = pd.read_csv(embedding_file_path)
    # string으로 저장된 embedding을 list로 변환
    df['embedding'] = df['embedding'].apply(ast.literal_eval)

    # embedding을 numpy array로 변환
    embeddings = df['embedding'].to_list()
    # faiss 인덱스 생성
    create_and_save_faiss_index(embeddings, index_path)

else:
    print(f"{embedding_file} is not exist")
