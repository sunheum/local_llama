import ast
from pathlib import Path

import openai
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI()

# openai embedding함수
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

# 파일명 입력 함수
def get_filename():
    filename = input("저장할 파일명을 입력하세요 (예: embedding.csv): ")
    return filename

# textspliter 설정
text_splitter = RecursiveCharacterTextSplitter(
    # 청크 크기 설정(약 500글자)
    chunk_size=250,
    # 청크 간의 중복되는 문자 수를 설정
    chunk_overlap=50,
    # 문자열 길이를 계산하는 함수를 지정합니다.
    length_function=len,
    # 구분자로 정규식을 사용할지 여부를 설정합니다.
    is_separator_regex=False,
)

folder_path = './database'
images_path = os.path.join(folder_path, 'images')
tables_path = os.path.join(folder_path, 'tables')
Path(folder_path).mkdir(parents=True, exist_ok=True)
Path(images_path).mkdir(parents=True, exist_ok=True)
Path(tables_path).mkdir(parents=True, exist_ok=True)

embedding_file = get_filename()
embedding_file_path = os.path.join(folder_path, embedding_file)

# embedding.csv가 존재하면, 데이터프레임 df로 로드
if os.path.exists(embedding_file_path):
    print(f"{embedding_file} is exist")
    df = pd.read_csv(embedding_file_path)
    # string으로 저장된 embedding을 list로 변환
    df['embedding'] = df['embedding'].apply(ast.literal_eval)
# pdf파일을 읽어서 embedding하여 csv파일로 저장
else:
    dataset_path = './guide_data'
    pdf_files = [file for file in os.listdir(dataset_path) if file.endswith('.pdf')]

    data = []
    # 모든 PDF 파일을 순회하며 embedding
    for file in pdf_files:
        pdf_file_path = os.path.join(dataset_path, file)
        reader = pdfplumber.open(pdf_file_path)

        # 각 pdf의 페이지별로 embedding
        for i, page in enumerate(reader.pages):
            # 이미지 추출 및 경로 기록
            image_paths = ';'.join([os.path.join(images_path, f"{Path(file).stem}_page_{i}_image_{image_index}.png")
                                    for image_index, image in enumerate(page.images)])
            # 이미지 저장
            for image_index, image_dict in enumerate(page.images):
                boxpoint = (image_dict['x0'], max(image_dict['top'],0), image_dict['x1'], image_dict['bottom'])
                copy_crop = page.crop(boxpoint)
                image_data = copy_crop.to_image(resolution=400)
                image_filename = f"{Path(file).stem}_page_{i}_image_{image_index}.png"
                image_file_path = os.path.join(images_path, image_filename)
                image_data.save(image_file_path)

            # 테이블 추출 및 경로 기록
            table_paths = ';'.join([os.path.join(tables_path, f"{Path(file).stem}_page_{i}_table_{table_index}.csv")
                                    for table_index, _ in enumerate(page.extract_tables())])
            # 테이블 저장
            for table_index, table in enumerate(page.extract_tables()):
                table_filename = f"{Path(file).stem}_page_{i}_table_{table_index}.csv"
                table_file_path = os.path.join(tables_path, table_filename)
                pd.DataFrame(table).to_csv(table_file_path, index=False, header=False)

            # 텍스트 추출
            text = page.extract_text() if page.extract_text() else ''  # 있을 경우 추출, 없으면 빈문자열
            # 텍스트 분할
            chunks = text_splitter.split_text(text)
            for chunk in chunks:
                data.append([file, i, chunk, image_paths, table_paths])  # 파일명, 페이지번호, 텍스트


    # 데이터프레임으로 생성
    # df = pd.DataFrame(data, columns=['filename', 'page', 'text'])
    df = pd.DataFrame(data, columns=['filename', 'page', 'text', 'image_paths', 'table_paths'])

    # embedding
    df['embedding'] = df['text'].apply(lambda x: get_embedding(x, model="text-embedding-3-small"))

    # csv파일로 저장
    df.to_csv(embedding_file_path, index=False, encoding='utf-8')
