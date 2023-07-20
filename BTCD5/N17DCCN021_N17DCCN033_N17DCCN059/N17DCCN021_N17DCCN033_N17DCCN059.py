# BÀI TẬP CHUYÊN ĐỀ 5
# Sinh viên thực hiện:
# MSV: N17DCCN021 - HỌ TÊN: NGUYỄN ANH DŨNG
# MSV: N17DCCN033 - HỌ TÊN: LÊ PHƯỚC ANH ĐẠT
# MSV: N17DCCN059 - HỌ TÊN: TRỊNH ĐỨC HUY
import math
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
# print(docs)
# for rank, index in enumerate(docs):
#     print(f"Rank {rank+1}: Document {index} ")

# Khởi tạo Inverted Index
inverted_index = {}
# Xây dựng Inverted Index
for i in range(0, len(docs)):
    words = docs[i].lower().split()
    words = set(words)
    for word in words:
        if word not in inverted_index:
            inverted_index[word] = []
        inverted_index[word].append(i + 1)

# Inverted Index sẽ có dạng {'từ khóa': {tài liệu 1, tài liệu 2, ...}}
# Sort the inverted index by the values of the sets of document IDs
inverted_index = dict(dict(sorted(inverted_index.items())))
print(inverted_index)
for key, value in inverted_index.items():
    print(f"{key:<15s} {len(value):<5d}: {value}")

# Truy vấn
query = "MEASUREMENT OF DIELECTRIC CONSTANT OF LIQUIDS BY THE USE OF MICROWAVE TECHNIQUES"
# query = "computer electronic system"
print(f"Câu truy vấn ban đầu: {query}")
# Tokenize the input text
tokens = nltk.word_tokenize(query)

# Remove the stop words from the tokens
filtered_tokens = [token for token in tokens if token.lower() not in stop_words]

# Join the filtered tokens into a new text
new_query = " ".join(filtered_tokens)
print(f"Câu truy vấn được chuẩn hóa: {new_query}")
# Tách từ khóa trong truy vấn và tìm tài liệu chứa các từ khóa
keywords = new_query.lower().split()
print(f"Bộ từ trong câu truy vấn: {keywords}")

# số lượng tài liệu
N = len(docs)
print(f"Số lượng tài liệu: {N}")

# hàm chuẩn hóa
def tokenize(text):
    return text.lower().split()

# tách từ từ tài liệu
document_tokens = [tokenize(document) for document in docs]

# tính TF (Số lần từ t xuất hiện trong văn bản Doc)
def compute_tf(tokens):
    tf_values = {}
    total_tokens = len(tokens)
    for token in tokens:
        tf_values[token] = tf_values.get(token, 0) + 1 / total_tokens
    return tf_values

document_tfs = [compute_tf(tokens) for tokens in document_tokens]
query_tf = compute_tf(keywords)

# tính IDF (Đại lượng nghịch đảo của df)
def compute_idf(documents, query_tokens):
    idf_values = {}
    all_tokens = query_tokens.copy()
    for tokens in documents:
        all_tokens.extend(tokens)
    total_documents = len(documents) + 1
    for token in all_tokens:
        document_count = sum(1 for tokens in documents if token in tokens) + 1
        idf_values[token] = math.log(total_documents / document_count)
    return idf_values

idf_values = compute_idf(document_tokens, keywords)

# Tính TF-IDF
def compute_tfidf(tf, idf):
    tfidf_values = {}
    for token, tf_value in tf.items():
        tfidf_values[token] = tf_value * idf.get(token, 0)
    return tfidf_values

document_tfidfs = [compute_tfidf(tf, idf_values) for tf in document_tfs]
query_tfidf = compute_tfidf(query_tf, idf_values)
print(f"document_tfidfs: {document_tfidfs}")
print(f"query_tfidf: {query_tfidf}")
# chuẩn hóa cosine
def compute_cosine_similarity(vector1, vector2):
    dot_product = sum(vector1.get(token, 0) * vector2.get(token, 0) for token in set(vector1) & set(vector2))
    magnitude1 = math.sqrt(sum(value**2 for value in vector1.values()))
    magnitude2 = math.sqrt(sum(value**2 for value in vector2.values()))
    return dot_product / (magnitude1 * magnitude2)

cosine_similarities = [compute_cosine_similarity(query_tfidf, document_tfidf) for document_tfidf in document_tfidfs]
print(f"Tương đồng Cosine: {cosine_similarities}")
def find_common_elements(arr1, arr2):
    set1 = set(arr1)
    set2 = set(arr2)
    common_elements = set1.intersection(set2)
    return list(common_elements)

# tài liệu liên quan
relevant_documents = [docs[index] for index, score in enumerate(cosine_similarities) if score > 0]
relevant_document_ids = [index + 1 for index, score in enumerate(cosine_similarities) if score > 0]
# in các tài liệu liên quan
print(f"Tài liệu phù hợp: {len(relevant_documents)}")
print(f"relevant_documents: {relevant_documents}")
print(f"relevant_document_ids: {relevant_document_ids}")
print(f"document_tokens : {document_tokens}")
# tính điểm cho mỗi tài liệu liên quan
document_scores = []
for index, doc_tokens in enumerate(document_tokens):
    score = 0
    print(doc_tokens)
    # Compute RSV score
    # Số lượng tài liệu liên quan đến câu truy vấn
    S = len(relevant_documents)
    print(f"S: {S}")
    # Số lượng tài liệu
    N = len(docs)
    for query_token in keywords:
        if query_token in inverted_index:
            if query_token in doc_tokens:
                print(f"===============================")
                print(f"relevant_document_ids: {relevant_document_ids}")
                # Số lượng tài liệu của từ truy vấn liên quan đến câu truy vấn
                # s = len(relevant_document_ids & inverted_index[query_token])
                s = len(find_common_elements(relevant_document_ids, inverted_index[query_token]))
                print((f"query_token: {inverted_index[query_token]}"))
                print(f"s: {s}")
                # số lượng tài liệu chứa từ t
                dft = len(inverted_index[query_token])
                print(f"dft: {dft}")
                print(query_token)
                print(f"===============================")
                # ct là trọng số của mỗi từ trong tài liệu
                # ct được tính theo công thức
                # thêm 0.5 để làm mịn
                ct = math.log(((s + 0.5) / (S - s + 0.5)) / ((dft - s + 0.5) / (N - dft - S + s + 0.5)))
                # ct = math.log((N - dft + 0.5) / dft + 0.5)
                # score bằng tổng các ct
                score += ct
    document_scores.append((index, score))

# Sắp xếp tài liệu dựa trên điểm số
document_scores.sort(key=lambda x: x[1], reverse=True)

# Lưu các tài liệu liên quan với điểm số lớn hơn 0
relevant_documents = [(index, score) for index, score in document_scores if score > 0]

# In các tài liệu liên quan với điểm số
stt = 0
print()
print("*******************************************************************************")
print("Danh sách tài liệu liên quan:")
for index, score in relevant_documents:
    stt += 1
    print("STT: ", stt)
    print(f"Document {index+1}: {docs[index]}")
    print("Score:", score)
    print()