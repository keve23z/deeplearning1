from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Tải mô hình GPT-2 và tokenizer đã được huấn luyện sẵn từ thư mục "./model"
model = GPT2LMHeadModel.from_pretrained("./model")
tokenizer = GPT2Tokenizer.from_pretrained("./model")

# Đặt token padding là token kết thúc câu (eos_token)
tokenizer.pad_token = tokenizer.eos_token

def generate_answer(question):
    # Chuẩn bị đầu vào, định dạng câu hỏi theo dạng "Q: câu hỏi\nA:"
    input_text = f"Q: {question}\nA:"
    
    # Mã hóa đầu vào thành token (dưới dạng tensor của PyTorch)
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    
    # Sinh câu trả lời từ mô hình, với độ dài tối đa 256 token
    # pad_token_id được chỉ định là eos_token_id để đảm bảo việc padding sử dụng token kết thúc câu
    output = model.generate(input_ids, max_length=256, pad_token_id=tokenizer.eos_token_id)
    
    # Giải mã đầu ra từ các token thành chuỗi văn bản
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Trích xuất phần trả lời sau "A:" và loại bỏ khoảng trắng thừa
    return response.split("A:")[-1].strip()
