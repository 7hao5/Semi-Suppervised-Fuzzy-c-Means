import pandas as pd
import numpy as np

def create_matrix(x):

    file_path = "C:\\GR1\\S_nhan.csv"
    # Đọc dữ liệu từ file CSV
    df = pd.read_csv(file_path)

    # Xác định số bản ghi n và giá trị lớn nhất k
    n = len(df)
    k = df.max().max()

    # Lấy ngẫu nhiên x% bản ghi
    sample_size = int(n * (x / 100))
    sampled_indices = np.random.choice(df.index, size=sample_size, replace=False)
    sampled_data = df.loc[sampled_indices]

    # Tạo ma trận sample_size x (k+1) với giá trị ban đầu bằng 0
    matrix = np.zeros((n, k+1))

    # Điền giá trị vào ma trận
    for i, row in sampled_data.iterrows():
        for value in row:
            if pd.notna(value):  # Đảm bảo giá trị không phải NaN
                matrix[i, int(value)] = 0.51  # Đánh dấu cột tương ứng với giá trị bản ghi

    # Chuyển ma trận thành DataFrame và đặt tên các cột
    columns = [f'Cot {i}' for i in range(k + 1)]
    matrix_df = pd.DataFrame(matrix, columns=columns)

    matrix_df.to_csv("C:\\GR1\\S_matrix.csv", index=False, mode='w')
