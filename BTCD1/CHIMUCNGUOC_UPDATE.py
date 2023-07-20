
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
docs = docs[:-1]
# for rank, index in enumerate(docs):
#     print(f"Rank {rank+1}: Document {index} ")

######### GIAI ĐOẠN 3: CHỈ MỤC NGƯỢC #########
# Khởi tạo Inverted Index
inverted_index = {}

# Đọc các tài liệu vào từ một danh sách các chuỗi

# Xây dựng Inverted Index
for i in range(0, len(docs)):
    words = docs[i].lower().split()
    words = set(words)
    for word in words:
        if word not in inverted_index:
            inverted_index[word] = []
        inverted_index[word].append(i + 1)
        # inverted_index[word] = list(set(inverted_index[word]))


# Inverted Index sẽ có dạng {'từ khóa': {tài liệu 1, tài liệu 2, ...}}
# Sort the inverted index by the values of the sets of document IDs
inverted_index = dict(dict(sorted(inverted_index.items())))
# print(inverted_index)
words_in_docs = []
for key, value in inverted_index.items():
    print(f"{key:<15s} {len(value):<5d}: {value}")
    words_in_docs.append(key)
# Truy vấn
# query = "MEASUREMENT OF DIELECTRIC CONSTANT OF LIQUIDS BY THE USE OF MICROWAVE TECHNIQUES"
#query = 'electronic AND transformer '
query = 'system AND transformer AND vector'
# query = "system AND nana"
def intersection_with_documents(p1, p2):
    i, j = 0, 0
    result = []

    while (p1 is not None and i < len(p1)) and (p2 is not None and j < len(p2)):
        # Gán docId
        doc_id1 = p1[i]
        doc_id2 = p2[j]

        # Nếu docId trùng nhau thì lưu docId vào kết quả, và tăng vị trí so sánh lên
        if doc_id1 == doc_id2:
            result.append(doc_id1)
            i += 1
            j += 1
        elif doc_id1 < doc_id2:
            i += 1
        else:
            j += 1

    return result

def optimaize_query(query):
    result = {}
    terms = query.split(' ')
    for term in terms:
        if term == "AND":
            terms.remove(term)
    for term in terms:
        if term not in words_in_docs:
            terms.remove(term)
    print(terms)
    for word in terms:
        if word in inverted_index:
            result[word] = inverted_index[word]

    result = dict(sorted(result.items(), key=lambda x: len(x[1])))
    return result

query_keywords_sort = optimaize_query(query)
print("query_keywords_sort : ", query_keywords_sort)

def intersect_postings_lists(query, inverted_indexes):
    # tách từ truy vấn từ câu query
    terms = query.split(' ')
    for term in terms:
        if term == "AND":
            terms.remove(term)
    print(terms)
    new_terms = []
    for term in terms:
        if term in words_in_docs:
            new_terms.append(term)
    print(new_terms)
    if len(new_terms) == 0:
        return []
    if len(new_terms) == 1:
        return inverted_index[new_terms[0]]
    # lấy danh sách thẻ định vị của mỗi từ trong câu truy vấn
    postings_lists = [inverted_indexes[term] for term in new_terms]
    result = postings_lists[0]
    print("postings_lists: ", postings_lists)

    for postings_list in postings_lists[1:]:
        result = intersection_with_documents(result, postings_list)

    return result

def intersect_postings_lists_optimize(query,inverted_indexes):
    sort_by_df = {}
    terms = query.split(' ')
    for term in terms:
        if term == "AND":
            terms.remove(term)
    print(terms)
    new_terms = []
    for term in terms:
        if term in words_in_docs:
            new_terms.append(term)
    print(new_terms)
    if len(new_terms) == 0:
        return []
    if len(new_terms) == 1:
        return inverted_index[new_terms[0]]

    for word in new_terms:
        if word in inverted_indexes:
            sort_by_df[word] = inverted_indexes[word]
    # Sắp xếp tăng dần theo df
    sort_by_df = dict(sorted(sort_by_df.items(), key=lambda x: len(x[1])))
    print("Sắp xếp thuật ngữ tăng dần theo df(t):")
    print(sort_by_df)
    postings_lists = [sort_by_df[term] for term in sort_by_df]
    print("postings_lists_op: ", postings_lists)
    # Khởi tạo tập kết quả result là danh sách ngắn nhất
    result = postings_lists[0]
    for postings_list in postings_lists[1:]:
        result = intersection_with_documents(result, postings_list)
    return result


result = intersect_postings_lists(query, inverted_index)
result_op = intersect_postings_lists_optimize(query, inverted_index)
print("KẾT QUẢ:")
print("intersect_postings_lists: ", result)
print("intersect_postings_lists_optimize: ", result_op)



# print("KẾT QUẢ:")
# print(intersection)
