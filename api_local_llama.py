from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict

model_path = "/data/workspace/Llama-3-Alpha-Ko-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
)

app = FastAPI()

class ChatRequest(BaseModel):
    user_message: str

@app.post("/chat")
def generate_answer(req: ChatRequest) -> Dict[str, str]:
    user_input = req.user_message
    messages = [
        {"role": "system", "content": "당신은 인공지능 어시스턴트입니다. 묻는 말에 친절하고 정확하게 답변하세요."},
        {"role": "user", "content": "{}".format(user_input)},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=False,
        repetition_penalty=1.05,
    )
    response = outputs[0][input_ids.shape[-1]:]

    return {
        "answer": tokenizer.decode(response, skip_special_tokens=True)
    }