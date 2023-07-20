
import re
import nltk
from nltk.corpus import stopwords
import math

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

def intersection_with_skip_pointers(p1, p2):
    i, j = 0, 0
    # skip1, skip2 = int(math.sqrt(len(p1))), int(math.sqrt(len(p2)))
    skip1, skip2 = 2, 2
    result = []

    while (p1 is not None and i < len(p1)) and (p2 is not None and j < len(p2)):
        # Nếu phần tử tại chỉ số i trong p1 bằng phần tử tại chỉ số j trong p2
        if p1[i] == p2[j]:
            result.append(p1[i]) # Thêm phần tử vào kết quả
            i += 1
            j += 1

        # Nếu phần tử tại chỉ số i trong p1 nhỏ hơn phần tử tại chỉ số j trong p2
        elif p1[i] < p2[j]:
            # Nếu i + skip1 vẫn nằm trong giới hạn của p1 và phần tử tại i + skip1 nhỏ hơn hoặc bằng phần tử tại j trong p2
            if i + skip1 < len(p1) and p1[i + skip1] <= p2[j]:
                while i + skip1 < len(p1) and p1[i + skip1] <= p2[j]:
                    # Tiến hành nhảy về phía trước bằng skip1
                    i += skip1
            else:
                i += 1 # Ngược lại, tăng i lên 1 đơn vị
        else:
            # Nếu phần tử tại chỉ số i trong p1 lớn hơn phần tử tại chỉ số j trong p2
            # Nếu j + skip2 vẫn nằm trong giới hạn của p2 và phần tử tại j + skip2 nhỏ hơn hoặc bằng phần tử tại i trong p1
            if j + skip2 < len(p2) and p2[j + skip2] <= p1[i]:
                while j + skip2 < len(p2) and p2[j + skip2] <= p1[i]:
                    j += skip2 # Tiến hành nhảy về phía trước bằng skip2
            else:
                j += 1 # Ngược lại, tăng j lên 1 đơn vị

        # skip1 = int(math.sqrt(len(p1))) # Cập nhật skip1 về giá trị căn bậc hai của độ dài p1
        # skip2 = int(math.sqrt(len(p2))) # Cập nhật skip2 về giá trị căn bậc hai của độ dài p2

    return result

def intersect_postings_lists(query, inverted_indexes):
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
    postings_lists = [inverted_indexes[term] for term in new_terms]
    result = postings_lists[0]

    for postings_list in postings_lists[1:]:
        result = intersection_with_skip_pointers(result, postings_list)

    return result

result = intersect_postings_lists(query, inverted_index)
print("=================================================")
print("KẾT QUẢ:")
print(result)
