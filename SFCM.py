import math, random, operator
import time

import numpy as np

from Create_Matrix import *
from sklearn.metrics import pairwise_distances

# Maximum number of iterations
MAX_ITER = 10000
# Fuzzy parameter
m = 1.7

# khoi tao ma tran thanh vien
def initializeMembershipMatrix(n, k):  # initializing the membership matrix
    membership_mat = []
    for i in range(n):
        random_num_list = [random.random() for i in range(k)]
        summation = sum(random_num_list)
        temp_list = [x / summation for x in random_num_list]

        flag = temp_list.index(max(temp_list))
        for j in range(0, len(temp_list)):
            if (j == flag):
                temp_list[j] = 1
            else:
                temp_list[j] = 0

        membership_mat.append(temp_list)
    return membership_mat

# tinhs taam cum
def calculateClusterCenter(supervised_membership_matrix_list, membership_mat, dataset, n, k):

    cluster_mem_val = list(zip(*membership_mat))
    cluster_sup_val = list(zip(*supervised_membership_matrix_list))
    cluster_centers = []
    for j in range(k):
        x = list(cluster_mem_val[j])
        z = list(cluster_sup_val[j])
        xraised = [(abs(p - q)) ** m for p, q in zip(x, z)]
        denominator = sum(xraised)
        temp_num = []
        for i in range(n):
            data_point = list(dataset.iloc[i])
            prod = [xraised[i] * val for val in data_point]
            temp_num.append(prod)
        numerator = map(sum, list(zip(*temp_num)))
        center = [z/denominator for z in numerator]
        cluster_centers.append(center)
    return cluster_centers

# cap nhat gia tri thanh vien
def updateMembershipValue(supervised_membership_matrix_list, cluster_centers, n, dataset, k):
    # Khởi tạo một danh sách hai chiều rỗng với kích thước n x k
    arr = [[0.0 for _ in range(k)] for _ in range(n)]

    p = float(1/(m-1))
    for i in range(n):
        x = list(dataset.iloc[i])
        distances = [np.linalg.norm(np.array(list(map(operator.sub, x, cluster_centers[j])))) for j in range(k)]
        for j in range(k):
            den = sum([math.pow(float(1/(distances[c]**2)), p) for c in range(k)])
            heso = 1 - sum([supervised_membership_matrix_list[i][c] for c in range(k)])
            tu = math.pow(float(1/(distances[j]**2)), p)
            arr[i][j] = supervised_membership_matrix_list[i][j] + float(heso*tu/den)
    return arr

# lay cac tam cum
def getClusters(membership_mat, n): # getting the clusters
    cluster_labels = list()
    for i in range(n):
        max_val, idx = max((val, idx) for (idx, val) in enumerate(membership_mat[i]))
        cluster_labels.append(idx)
    return cluster_labels

def calculate_norm(U_new, U_old):
    norm = 0.0
    for i in range(len(U_new)):
        for j in range(len(U_new[i])):
            norm += (U_new[i][j] - U_old[i][j]) ** 2
    return np.sqrt(norm)

# thuat toan FCM voi tam cum ban dau la cac diem du lieu ngau nhien
def semiFuzzyCMeansClustering(h, k, epsilon):  # Third iteration Random vectors from data

    dataset = pd.read_csv("C:\\GR1\\S_data.csv")
    n = len(dataset)

    # Tạo ma trận thành viên bán giám sát
    create_matrix(h)
    combined_matrix = pd.read_csv("C:\\GR1\\S_matrix.csv")
    # Tìm vị trí các phần tử bằng 0.51 và tạo DataFrame
    indices = []

    for i in range(len(combined_matrix)):
        if (combined_matrix.iloc[i] == 0.51).any():
            indices.append(i)

    # Tạo DataFrame lưu chỉ số hàng
    index_df = pd.DataFrame(indices, columns=['index'])

    # Lưu DataFrame vào file CSV
    output_file_path = "C:\\GR1\\index.csv"
    index_df.to_csv(output_file_path, index=False, mode='w')

    # chuyển data Farme thành Lits
    combined_matrix_list = combined_matrix.values.tolist()

    # Ma tran thanh vien
    membership_mat = initializeMembershipMatrix(n, k)

    # ma tran thanh vien ban giam sat
    supervised_membership_matrix_list = combined_matrix_list
    curr = 0
    acc = []
    while True:
        cluster_centers = calculateClusterCenter(supervised_membership_matrix_list, membership_mat, dataset, n, k)
        membership_mat_new = updateMembershipValue(supervised_membership_matrix_list, cluster_centers,  n, dataset, k)
        cluster_labels = getClusters(membership_mat, n)

        acc.append(cluster_labels)
        if calculate_norm(membership_mat_new, membership_mat) < epsilon:
            break
        membership_mat = membership_mat_new
        curr += 1

    return cluster_labels, cluster_centers, acc, curr

def S_star(h, epsilon):

    k = 3

    start_time = time.time()
    labels, centers, acc, curr = semiFuzzyCMeansClustering(h, k, epsilon)
    end_time = time.time()
    print(curr)

    label = pd.DataFrame(labels, columns=['Label'])
    center = pd.DataFrame(centers, columns=['Column1', 'Column2', 'Column3', 'Column4'])
    label.to_csv("C:\\GR1\\S_predict.csv", index=False, mode='w')
    center.to_csv("C:\\GR1\\S_tam.csv", index=False, mode='w')
    execution_time = end_time - start_time

    return execution_time

def rand_index():

    label = pd.read_csv("C:\\GR1\\S_nhan.csv")
    p_label = pd.read_csv("C:\\GR1\\S_predict.csv")
    n = len(label)

    a = b = c = d = 0

    for i in range(n-1):
        for j in range(i+1, n):

            if label.iloc[i, 0] == label.iloc[j, 0] and p_label.iloc[i, 0] == p_label.iloc[j, 0]:
                a = a+1
            if label.iloc[i, 0] == label.iloc[j, 0] and p_label.iloc[i, 0] != p_label.iloc[j, 0]:
                b = b+1
            if label.iloc[i, 0] != label.iloc[j, 0] and p_label.iloc[i, 0] == p_label.iloc[j, 0]:
                c = c+1
            if label.iloc[i, 0] != label.iloc[j, 0] and p_label.iloc[i, 0] != p_label.iloc[j, 0]:
                d = d+1

    rand_in = (a+d)/(a+b+c+d)
    return rand_in

def dongBo_Cum_Lop():

    label = pd.read_csv("C:\\GR1\\S_nhan.csv")
    p_label = pd.read_csv("C:\\GR1\\S_predict.csv")
    n = len(label)

    rows = columns = 3
    array = np.zeros((rows, columns))

    for i in range(n):
        x = label.iloc[i, 0]
        y = p_label.iloc[i, 0]
        array[x][y] += 1

    # Tìm vị trí của phần tử lớn nhất trong mỗi hàng
    max_positions = []
    for row in array:
        max_index = np.argmax(row)
        max_positions.append(max_index)

    # Cập nhật giá trị trong cột 'Lable' của p_label
    for i in range(n):
        if p_label.iloc[i, 0] == max_positions[0]:
            p_label.at[i, 'Label'] = 0
        elif p_label.iloc[i, 0] == max_positions[1]:
            p_label.at[i, 'Label'] = 1
        elif p_label.iloc[i, 0] == max_positions[2]:
            p_label.at[i, 'Label'] = 2

    p_label.to_csv("C:\\GR1\\S_predict.csv", index=False, mode='w')


def accuracy():

    label = pd.read_csv("C:\\GR1\\S_nhan.csv")
    p_label = pd.read_csv("C:\\GR1\\S_predict.csv")
    n = len(label)

    t = 0
    for i in range(n):
        if label.iloc[i, 0] == p_label.iloc[i, 0]:
            t = t + 1

    return t/n

# tinh precision cua class x
def precision(x):

    label = pd.read_csv("C:\\GR1\\S_nhan.csv")
    p_label = pd.read_csv("C:\\GR1\\S_predict.csv")
    n = len(label)

    tp = t = 0
    for i in range(n):
        if p_label.iloc[i, 0] == x and label.iloc[i, 0] == x:
            tp = tp + 1
        if p_label.iloc[i, 0] == x:
            t = t + 1
    return tp/t

def recall(x):

    label = pd.read_csv("C:\\GR1\\S_nhan.csv")
    p_label = pd.read_csv("C:\\GR1\\S_predict.csv")
    n = len(label)

    tp = t = 0
    for i in range(n):
        if p_label.iloc[i, 0] == x and label.iloc[i, 0] == x:
            tp = tp + 1
        if label.iloc[i, 0] == x:
            t = t + 1
    return tp/t

def F1(x):

    pre = precision(x)
    re = recall(x)

    f1 = (2*pre*re)/(pre+re)

    return f1

def S_CHI():
    file_path = "C:\\GR1\\S_data.csv"
    dataset1 = pd.read_csv(file_path)
    labels = pd.read_csv("C:\\GR1\\S_predict.csv")
    centers = pd.read_csv("C:\\GR1\\S_tam.csv")
    centers = np.array(centers)
    labels = np.array(labels)

    # Đảm bảo các cột trong DataFrame chỉ chứa giá trị số
    dataset1 = dataset1.apply(pd.to_numeric, errors='coerce')

    # Tính trung bình của tất cả các cột
    means = dataset1.mean().values
    k = 3

    # Tính BCSS
    BCSS = 0
    for i in np.arange(k):
        cumi = 0
        for j in labels:
            if j == i:
                cumi = cumi + 1

        BCSS = BCSS + cumi * np.sum((centers[i] - means) ** 2)


    # Tính WCSS
    WCSS = 0
    for i in np.arange(k):
        idx_i = np.where(np.array(labels) == i)[0]
        if len(idx_i) == 0:
            continue
        WCSS_i = np.sum((dataset1.iloc[idx_i].values - centers[i]) ** 2)
        WCSS += WCSS_i

    # Tính chỉ số CH
    CH = (BCSS / (k - 1)) / (WCSS / (len(dataset1) - k))

    return CH


def S_DBI():

    dataset1 = pd.read_csv("C:\\GR1\\S_data.csv")
    X = dataset1.values

    nhan = pd.read_csv("C:\\GR1\\S_predict.csv")
    labels = nhan['Label'].values
    n_clusters = len(np.unique(labels))
    cluster_k = [X[labels == k] for k in range(n_clusters)]
    centroids = [np.mean(cluster, axis=0) for cluster in cluster_k]

    S = [np.mean(np.linalg.norm(cluster - centroid, axis=1)) for cluster, centroid in zip(cluster_k, centroids)]

    M = pairwise_distances(centroids, metric='euclidean')

    R = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):

            if i == j:
                R[i, j] = 0
            else:
                R[i, j] = (S[i] + S[j]) / M[i, j]

    D = []
    for i in range(n_clusters):
        D.append(np.max(R[i]))

    dbi = np.mean(D)
    return dbi

