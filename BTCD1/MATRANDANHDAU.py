
import re
import nltk
from nltk.corpus import stopwords
import numpy as np
from bitarray import bitarray

import bitarray
import json
import time
import pickle

nltk.download('punkt')
nltk.download('stopwords')

# Tải stop words
stop_words = set(stopwords.words('english'))


# Loại bỏ stop words
def remove_stopwords(documents):
    filtered_documents = []
    for doc in documents:
        words = nltk.word_tokenize(doc)
        filtered_words = [word for word in words if word.lower() not in stop_words]
        filtered_documents.append(' '.join(filtered_words))
    return filtered_documents

######### GIAI ĐOẠN 1: CHUẨN HÓA INPUT #########
# Mục tiêu: Xóa bỏ kí tự khoảng trắng thừa, kí tự đặc biệt, số,
# Biến đổi lowercase
def read_file_and_modify(filename):
    with open(filename, 'r') as f:
        text = f.read()
    clean_text = ""
    for char in text:
        if char not in "\n":
            clean_text += char
    text = text.strip()  # loại bỏ khoảng trắng thừa ở đầu và cuối văn bản
    words = text.split()  # tách văn bản thành các từ và loại bỏ khoảng trắng thừa giữa
    # các từ
    words = " ".join(words)
    clean_text = re.sub(r'\d+', '', words)  # loại bỏ số

    # loại bỏ khoảng trắng thừa ở đầu và cuối câu
    # Tách từng văn bản thành list
    clean_text = clean_text.split("/")
    for i in range(len(clean_text)):
        clean_text[i] = clean_text[i].strip().lower()
    return clean_text


######### GIAI ĐOẠN 2: CHUẨN HÓA DỮ LIỆU #########
docs = read_file_and_modify("doc.txt")
docs = remove_stopwords(docs)
# docs = docs[:-1]
# lưu danh sách tài liệu
list_docs = []
for i, doc in enumerate(docs):
    list_docs.append(i+1)

print(docs)
# tạo chỉ mục
term_document_matrix = {}

for i in range(0, len(docs)):
    words = docs[i].lower().split()
    words = set(words)
    for word in words:
        if word not in term_document_matrix:
            #Khởi tạo tập danh sách tài liệu chứa từ vựng: "từ vựng" : {ds tài liệu}
            term_document_matrix[word] = []
        term_document_matrix[word].append(i + 1)


# print(term_document_matrix)
term_document_matrix = dict(dict(sorted(term_document_matrix.items())))
print(term_document_matrix)
# in chỉ mục
for key, value in term_document_matrix.items():
    print(f"{key:<20s} : {value}")


# khởi tạo lưu trữ ma trận đánh dấu
matrix = {}

for key, value in term_document_matrix.items():
    matrix[key] = []
    for i in list_docs:
        if i in value:
            matrix[key].append(1)
        else:
            matrix[key].append(0)

# in ma trận đánh dấu
print("MA TRẬN ĐÁNH DẤU: ")
print(matrix)
for key, value in matrix.items():
    print(f"{key:<20s} : {value}")

def boolean_query(query, term_document_matrix):
    # Chia query thành các từ khóa
    keywords = query.split()
    print(keywords)
    result = []
    # Khởi tạo tập kết quả là tất cả các tài liệu
    if keywords[0] not in ['AND', 'OR', 'NOT']:
        result = matrix[keywords[0]]
    else:
        for i in list_docs:
            result.append(1)
    print(f"mảng: {result}")
    i = 0 #từ đầu tiên trong cấu truy vấn
    # Áp dụng các phép toán truy vấn
    while i < len(keywords):
        # gán từ cần truy vấn vào biến tạm
        term = keywords[i]
        if term not in ['AND', 'OR', 'NOT']:
            result = matrix[keywords[i]]
            print(f"====: {result}")
        # nếu gặp toán tử logic thì nhảy dến từ tiếp theo
        if term == 'AND':
            i += 1
            next_term = keywords[i]
            print(f"next_term: {matrix[next_term]}")
            print(f"mảng AND: {result}")
            result = [a and b for a, b in zip(result, matrix[next_term])]
            print(f"result: {result}")
        elif term == 'OR':
            i += 1
            next_term = keywords[i]
            print(next_term)
            print(f"mảng OR: {matrix[next_term]}")
            result = [a or b for a, b in zip(result, matrix[next_term])]
            print(f"result: {result}")
        elif term == 'NOT':
            i += 1
            next_term = keywords[i]
            print(next_term)
            print(f"mảng NOT: {matrix[next_term]}")
            result = [int(a and not b) for a, b in zip(result, matrix[next_term])]
            print(f"result: {result}")
        i += 1
    return result

# Truy vấn
# query = 'system AND transformer OR transistor NOT vector'
query = 'transformer OR transistor'
print(query)
# Thực hiện truy vấn và in kết quả
result = boolean_query(query, term_document_matrix)
print("Kết quả truy vấn:", result)
docs_result = []
for i, value in enumerate(result):
    if value == 1:
        docs_result.append(i+1)
        print(f"Tài liệu {i+1}: {docs[i]}")

print(f"Tài liệu cần tìm: {docs_result}")

