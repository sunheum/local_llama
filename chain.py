import os

from rag import query_csv

def read_prompt_template(file_path: str) -> str:
    with open(file_path, "r") as f:
        prompt_template = f.read()

    return prompt_template

def create_prompt(query):
    # 질문과 가장 관련있는 본문 3개 가져옴
    result = query_csv(query, use_retriever=True)

    references_source = '\n\n'.join([result.iloc[0]['filename'],
                                     result.iloc[2]['filename'],
                                     result.iloc[1]['filename']])
    references_text = '\n\n'.join([result.iloc[0]['text'],
                                   result.iloc[2]['text'],
                                   result.iloc[1]['text']])

    # 'page' 열의 값을 문자열로 변환
    result['page'] = result['page'].astype(str)
    references_text_page = '\n\n'.join([result.iloc[0]['page'],
                                   result.iloc[2]['page'],
                                   result.iloc[1]['page']])

    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    SYSTEM_MESSAGE_TEMPLATE = os.path.join(CUR_DIR, "prompt_templates", "system_message.txt")
    system_message = read_prompt_template(SYSTEM_MESSAGE_TEMPLATE)

    filled_system_message = system_message.format(document1=references_text.split('\n\n')[0],
                                                  document2=references_text.split('\n\n')[1],
                                                  document3=references_text.split('\n\n')[2])

    return filled_system_message, references_source, references_text, references_text_page