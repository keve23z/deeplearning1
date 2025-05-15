import pandas as pd

# Đọc file CSV vào DataFrame
df = pd.read_csv('chatbot.csv')

# Kiểm tra đầu dữ liệu
print(df.head())

# Chuẩn bị dữ liệu huấn luyện (cặp câu hỏi và câu trả lời)
train_data = []

# Duyệt qua từng dòng dữ liệu và chuẩn bị các câu hỏi và câu trả lời cho huấn luyện
for index, row in df.iterrows():
    question = row['Question']
    answer = row['Answer']
    train_data.append(f"Q: {question}\nA: {answer}\n")

# Lưu dữ liệu đã chuẩn bị vào file văn bản
with open('train_data.txt', 'w', encoding='utf-8') as file:
    file.writelines(train_data)


print("Dữ liệu đã được tiền xử lý và lưu vào 'train_data.txt'")
