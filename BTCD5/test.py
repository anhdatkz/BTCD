#
# import re
# import nltk
# from nltk.corpus import stopwords
# import numpy as np
# from bitarray import bitarray
#
# import bitarray
# import json
# import time
# import pickle
#
# nltk.download('punkt')
# nltk.download('stopwords')
#
# # Tải stop words
# stop_words = set(stopwords.words('english'))
#
#
# # Loại bỏ stop words
# def remove_stopwords(documents):
#     filtered_documents = []
#     for doc in documents:
#         words = nltk.word_tokenize(doc)
#         filtered_words = [word for word in words if word.lower() not in stop_words]
#         filtered_documents.append(' '.join(filtered_words))
#     return filtered_documents
#
# ######### GIAI ĐOẠN 1: CHUẨN HÓA INPUT #########
# # Mục tiêu: Xóa bỏ kí tự khoảng trắng thừa, kí tự đặc biệt, số,
# # Biến đổi lowercase
# def read_file_and_modify(filename):
#     with open('./doc-text', 'r') as f:
#         text = f.read()
#     clean_text = ""
#     for char in text:
#         if char not in "\n":
#             clean_text += char
#     text = text.strip()  # loại bỏ khoảng trắng thừa ở đầu và cuối văn bản
#     words = text.split()  # tách văn bản thành các từ và loại bỏ khoảng trắng thừa giữa
#     # các từ
#     words = " ".join(words)
#     clean_text = re.sub(r'\d+', '', words)  # loại bỏ số
#
#     # loại bỏ khoảng trắng thừa ở đầu và cuối câu
#     # Tách từng văn bản thành list
#     clean_text = clean_text.split("/")
#     for i in range(len(clean_text)):
#         clean_text[i] = clean_text[i].strip().lower()
#     return clean_text
#
#
# ######### GIAI ĐOẠN 2: CHUẨN HÓA DỮ LIỆU #########
# docs = read_file_and_modify("doc-text")
# docs = remove_stopwords(docs)
# docs = docs[:-1]
# print(docs)
# # for rank, index in enumerate(docs):
# #     print(f"Rank {rank+1}: Document {index} ")
#
# ######### GIAI ĐOẠN 3: CHỈ MỤC NGƯỢC #########
# # Khởi tạo Inverted Index
# inverted_index = {}
#
# # Đọc các tài liệu vào từ một danh sách các chuỗi
#
# # Xây dựng Inverted Index với bước nhảy step=2
# skip = 1
# for i in range(0, len(docs)):
#     words = docs[i].lower().split()
#     for word in words:
#         if word not in inverted_index:
#             inverted_index[word] = set()
#         inverted_index[word].add(i + 1)
#
# # Inverted Index sẽ có dạng {'từ khóa': {tài liệu 1, tài liệu 2, ...}}
# # Sort the inverted index by the values of the sets of document IDs
# inverted_index = dict(dict(sorted(inverted_index.items())))
# # print(inverted_index)
#
# # Truy vấn
# #query = "MEASUREMENT OF DIELECTRIC CONSTANT OF LIQUIDS BY THE USE OF MICROWAVE TECHNIQUES"
# query = "MEASUREMENT OF DIELECTRIC"
# # Tokenize the input text
# tokens = nltk.word_tokenize(query)
#
# # Remove the stop words from the tokens
# filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
#
# # Join the filtered tokens into a new text
# new_query = " ".join(filtered_tokens)
# print(new_query)
# # Tách từ khóa trong truy vấn và tìm tài liệu chứa các từ khóa
# keywords = new_query.lower().split()
# print(keywords)
# result = set()
#
# # Tìm kiếm các tài liệu chứa từ đầu tiên trong truy vấn
# term = keywords[0]
# doc_ids = inverted_index.get(term, set())
# for doc_id in doc_ids:
#     # Kiểm tra các từ còn lại trong truy vấn
#     match = True
#     for term in keywords[1:]:
#         # Nếu vị trí của tài liệu không phù hợp với bước nhảy, bỏ qua tài liệu đó
#         if doc_id not in inverted_index.get(term, set()):
#             match = False
#             break
#     # Nếu tất cả các từ trong truy vấn khớp với tài liệu hiện tại, thêm tài liệu đó vào kết quả
#     if match:
#         result.add(doc_id)
#     # Di chuyển đến tài liệu tiếp theo với bước nhảy
#     doc_id += skip
#     if doc_id not in doc_ids:
#         break
#
# # for keyword in keywords:
# #     # if keyword not in inverted_index:
# #     #     result = inverted_index[keyword]
# #     #     break
# #     # if result is None:
# #     result = set.intersection(*[inverted_index.get(term, set()) for term in keywords])
# #     #result = inverted_index[keyword]
# #     # else:
# #     #     # Lấy giao
# #     #     result = result.intersection(inverted_index[keyword])
#
# # Kết quả tìm được là tập hợp các chỉ số tài liệu chứa các từ khóa trong truy vấn
# # print("KẾT QUẢ")
# # print(result)
#
# # inverted_index = {
# #     'apple': {1, 2, 3},
# #     'banana': {2, 3, 4},
# #     'orange': {3, 4, 5},
# #     'pear': {4, 5, 6}
# # }
# #
# # query = "yellow"
# # terms = query.lower().split()
# # result = set.intersection(*[inverted_index.get(term, set()) for term in terms])
# # print(result)
import math

# Sample document collection
documents = [
    "This is the first document",
    "This document is the second document",
    "And this is the third one",
    "Is this the first document"
]

# Preprocessing step to tokenize the documents
tokenized_documents = [doc.lower().split() for doc in documents]

# Build the term-document matrix
term_document_matrix = {}
for doc_idx, doc in enumerate(tokenized_documents):
    for term in doc:
        if term not in term_document_matrix:
            term_document_matrix[term] = [0] * len(documents)
        term_document_matrix[term][doc_idx] = 1

# Calculate term frequencies and document frequencies
term_frequencies = {}
document_frequencies = {}
for term, doc_vector in term_document_matrix.items():
    term_frequencies[term] = sum(doc_vector)
    document_frequencies[term] = sum([1 for val in doc_vector if val > 0])


# Function to calculate relevance score
def calculate_relevance_score(query):
    query_terms = query.lower().split()
    relevance_scores = [0] * len(documents)

    for term in query_terms:
        if term not in term_document_matrix:
            continue

        term_probability = term_frequencies[term] / len(documents)
        term_inverse_probability = 1 - term_probability

        for doc_idx, doc_vector in enumerate(term_document_matrix[term]):
            if doc_vector == 1:
                relevance_scores[doc_idx] += math.log(term_probability / term_inverse_probability)

    return relevance_scores


# Example usage
query = "first document"
scores = calculate_relevance_score(query)
ranked_documents = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

# Print the ranked documents
for doc, score in ranked_documents:
    print(f"Document: {doc}\nRelevance Score: {score}\n")
