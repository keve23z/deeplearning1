import pandas as pd
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch
from transformers import TrainerCallback
import os
import glob

# === 1. Load danh sách câu hỏi đã huấn luyện (nếu file tồn tại) ===
trained_questions_file = "trained_questions.txt"
if os.path.exists(trained_questions_file):
    with open(trained_questions_file, "r", encoding="utf-8") as f:
        trained_questions = set(line.strip() for line in f.readlines())
else:
    trained_questions = set()

# === 2. Đọc và xử lý dữ liệu từ file CSV ===
df = pd.read_csv("chatbot.csv", encoding="utf-8", on_bad_lines='skip')
df = df[['Question', 'Answer']].dropna()

# === 3. Lọc ra các câu hỏi chưa được huấn luyện ===
new_data = df[~df['Question'].isin(trained_questions)]

if new_data.empty:
    print("Không có dữ liệu mới để huấn luyện.")
    exit()

# === 4. Tạo cột 'text' định dạng cho GPT-2 ===
new_data['text'] = "Q: " + new_data['Question'] + "\nA: " + new_data['Answer']

# === 5. Chuyển DataFrame sang Dataset của HuggingFace ===
dataset = Dataset.from_pandas(new_data[['text']])

# === 6. Tokenizer ===
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize(sample):
    return tokenizer(sample['text'], padding="max_length", truncation=True, max_length=256)

tokenized_data = dataset.map(tokenize, batched=True)

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
