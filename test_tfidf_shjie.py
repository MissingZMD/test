def get_data():
    with open('../prepare/train_data/词性标记完数据.txt',encoding='utf-8') as f:
    #with open('../prepare/pos_data/p_data.txt', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [i.strip() for i in lines]
        print(lines)
    return lines


def to_tfidf(data):
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfv = TfidfVectorizer()
    tfv.fit(data)
    data_tf = tfv.transform(data).toarray()
    print(tfv.get_feature_names())
    for index,em in enumerate(data_tf[0]):
        print(index,em)
    print(tfv.get_feature_names()[0])
    print(data[1])


if __name__ == '__main__':
    lines = get_data()
    to_tfidf(lines)

