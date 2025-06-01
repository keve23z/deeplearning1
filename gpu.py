# Import thư viện TensorFlow và NumPy
import tensorflow as tf
import numpy as np

# Đặt seed cho NumPy và TensorFlow để kết quả có thể tái lập (reproducible)
np.random.seed(0)
tf.random.set_seed(0)

# In ra số lượng GPU vật lý đang được nhận diện
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Cố gắng cấu hình để chỉ sử dụng GPU đầu tiên nếu có
try:
    gpus = tf.config.list_physical_devices('GPU')  # Lấy danh sách các GPU vật lý
    if gpus:
        # Cấu hình để chỉ GPU đầu tiên được sử dụng (hữu ích khi có nhiều GPU)
        tf.config.set_visible_devices(gpus[0], 'GPU')

        # Lấy danh sách các GPU logic (có thể chia nhỏ từ GPU vật lý)
        logical_gpus = tf.config.list_logical_devices('GPU')

        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        print("Successfully set visible devices to GPU 0")
except RuntimeError as e:
    # Nếu có lỗi khi cấu hình GPU (ví dụ cấu hình sau khi khởi tạo), in lỗi ra
    print(e)
