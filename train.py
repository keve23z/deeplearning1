# === Import và cấu hình ===

import pandas as pd
import torch
import os
import glob
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)

# === 1. Load danh sách câu hỏi đã huấn luyện (nếu file tồn tại) ===
# === Load danh sách câu hỏi đã huấn luyện (nếu có) ===
trained_questions_file = "trained_questions.txt"
if os.path.exists(trained_questions_file):
    with open(trained_questions_file, "r", encoding="utf-8") as f:
        trained_questions = set(line.strip() for line in f.readlines())
else:
    trained_questions = set()

# === 2. Đọc và xử lý dữ liệu từ file txt===
# === Đọc dữ liệu từ file ===
train_data_file = "train_data.txt"
questions, answers = [], []

with open(train_data_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

current_question = None
for line in lines:
    line = line.strip()
    if line.startswith("Q:"):
        current_question = line[2:].strip()
    elif line.startswith("A:") and current_question:
        current_answer = line[2:].strip()
        questions.append(current_question)
        answers.append(current_answer)
        current_question = None

# === 3. Lọc ra các câu hỏi chưa được huấn luyện ===
df = pd.DataFrame({"Question": questions, "Answer": answers})
new_data = df[~df['Question'].isin(trained_questions)].copy()

# === 4. Tạo cột 'text' định dạng cho GPT-2 ===
if new_data.empty:
    print("Không có dữ liệu mới để huấn luyện.")
else:
    new_data['text'] = "Q: " + new_data['Question'] + "\nA: " + new_data['Answer']
    print(f"Đã lọc {len(new_data)} câu hỏi mới.")


# === 5.Chuẩn bị Dataset và Tokenizer===
# ===Chuyển DataFrame sang Dataset của HuggingFace ===
# ✅ Kiểm tra và chuẩn hóa dữ liệu
new_data = new_data.dropna(subset=["text"])
new_data = new_data[new_data["text"].str.strip().astype(bool)]
dataset = Dataset.from_pandas(new_data[["text"]])

# ✅ Chia train/test 85% train và 15% validation
split_dataset = dataset.train_test_split(test_size=0.15, seed=42)


# === 6. Tokenizer ===
# ✅ Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize(sample):
    outputs = tokenizer(sample["text"], padding="max_length", truncation=True, max_length=256)
    outputs["labels"] = outputs["input_ids"].copy()  # <-- Thêm dòng này
    return outputs


tokenized_data = split_dataset.map(tokenize, batched=True, remove_columns=["text"])

# ✅ In lại số lượng
print("Train samples:", len(tokenized_data["train"]))
print("Test samples :", len(tokenized_data["test"]))

# === 7. Load model GPT-2 ===
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

# === 8. Callback in tiến độ ===
class ProgressCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.max_steps is not None:
            percent_complete = 100 * state.global_step / state.max_steps
            print(f"Đã huấn luyện: {percent_complete:.2f}% ({state.global_step}/{state.max_steps} bước)")

# === 9. Thiết lập tham số huấn luyện ===
args = TrainingArguments(
    output_dir="./model",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_dir="./logs",
    save_steps=500,
    logging_steps=100,
    save_total_limit=2  # Giữ lại tối đa 2 checkpoint
)

# === 10. Tạo Trainer ===
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_data,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    callbacks=[ProgressCallback()]
)

# === 11. Kiểm tra checkpoint gần nhất (nếu có) ===
checkpoints = sorted(glob.glob("./model/checkpoint-*"), key=os.path.getmtime)
checkpoint_path = checkpoints[-1] if checkpoints else None

# === 12. Huấn luyện mô hình (có thể tiếp tục từ checkpoint) ===
trainer.train(resume_from_checkpoint=checkpoint_path)

# === 13. Lưu mô hình cuối cùng ===  
trainer.save_model("./model")

# === 14. Ghi lại các câu hỏi đã huấn luyện ===
with open(trained_questions_file, "a", encoding="utf-8") as f:
    for q in new_data['Question']:
        f.write(q.strip() + "\n")
