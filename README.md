# Llama 로컬 모델

- 사용 모델 : Llama-3-Alpha-Ko-8B-Instruct
- 실행 명령어
  - nohup streamlit run app.py > app.log 2>&1 &
  - nohup streamlit run chatlog.py > chat.log 2>&1 &
  - nohup uvicorn api:app --host 0.0.0.0 --port 8080 > fastapi.log 2>&1 &
