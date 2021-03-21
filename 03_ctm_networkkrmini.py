# -*- coding: utf-8 -*-
'''
tomotopy 
This example shows how to perform a Correlated Topic Model using tomotopy 
and visualize the correlation between topics.
Required Packages:
    nltk, sklearn, pyvis
'''

import tomotopy as tp
import nltk
from nltk.corpus import stopwords
import re
from sklearn.datasets import fetch_20newsgroups
from pyvis.network import Network
import re # 정규표현식 패키지
import tomotopy as tp # 토픽 모델링에 사용할 패키지
from kiwipiepy import Kiwi # 한국어 형태소 분석에 사용할 패키지
from pyvis.network import Network # 네트워크 시각화에 사용할 패키지

try:
    #raise EnvironmentError
    # load if preprocessed corpus exists
    corpus = tp.utils.Corpus.load('input/input.cps')
    print("load_done")
except :
    kiwi = Kiwi()
    kiwi.prepare()
 
    # 형태소 분석 후 사용할 태그
    pat_tag = re.compile('NN[GP]|V[VA]|MAG|MM')
 
    def tokenizer(raw, user_data):
        res, _ = user_data()[0]
        for w, tag, start, l in res:
            if not pat_tag.match(tag) or len(w) <= 1: continue
            yield w + ('다' if tag.startswith('V') else ''), start, l

    corpus = tp.utils.Corpus(
        tokenizer=tokenizer
    )
    # 입력 파일에는 한 줄에 문헌 하나씩이 들어가 있습니다.
    corpus.process((line, kiwi.async_analyze(line)) for line in open('input/input.csv', encoding='utf-8'))
    # 전처리한 코퍼스를 저장한다.
    corpus.save('output/output.cps')
    
    # porter_stemmer = nltk.PorterStemmer().stem
    # english_stops = set(porter_stemmer(w) for w in stopwords.words('english'))
    # pat = re.compile('^[a-z]{2,}$')
    # corpus = tp.utils.Corpus(
    #     tokenizer=tp.utils.SimpleTokenizer(porter_stemmer), 
    #     stopwords=lambda x: x in english_stops or not pat.match(x)
    # )
    #corpus.process(open('이성사랑방_titlearticle.txt'))
#newsgroups_train = fetch_20newsgroups()
 #   corpus.process(d.lower() for d in newsgroups_train.data)
    # save preprocessed corpus for reuse
    #corpus.save('preprocessed_20news.cps')

# 최소 10개 이상 문헌에 등장하고, 전체 출현빈도는 20 이상인 단어만 사용합니다.
# 그리고 상위 10개 고빈도 단어는 분석에서 제외하구요
# 주제 개수는 40개입니다.
mdl = tp.CTModel(tw=tp.TermWeight.ONE, min_df=10, min_cf=20, rm_top=5, k=40, corpus=corpus)
mdl.train(0)
# mdl = tp.CTModel(tw=tp.TermWeight.IDF, min_df=5, rm_top=40, k=30, corpus=corpus)
# mdl.train(0)

# Since we have more than ten thousand of documents, 
# setting the `num_beta_sample` smaller value will not cause an inaccurate result.
mdl.num_beta_sample = 4
print('Num docs:{}, Num Vocabs:{}, Total Words:{}'.format(
    len(mdl.docs), len(mdl.used_vocabs), mdl.num_words
))
print('Removed Top words: ', mdl.removed_top_words)

# Let's train the model
for i in range(0, 60, 20):
    print('Iteration: {:04}, LL per word: {:.4}'.format(i, mdl.ll_per_word))
    mdl.train(20)
print('Iteration: {:04}, LL per word: {:.4}'.format(1000, mdl.ll_per_word))

mdl.summary()

# Let's visualize the result
g = Network(width=800, height=800, font_color="#333")
correl = mdl.get_correlations().reshape([-1])
correl.sort()
top_tenth = mdl.k * (mdl.k - 1) // 10
top_tenth = correl[-mdl.k - top_tenth]

for k in range(mdl.k):
    label = "#{}".format(k)
    title= ' '.join(word for word, _ in mdl.get_topic_words(k, top_n=8))
    print('Topic', label, title)
    g.add_node(k, label=label, title=title, shape='ellipse')
    for l, correlation in zip(range(k - 1), mdl.get_correlations(k)):
        if correlation < top_tenth: continue
        g.add_edge(k, l, value=float(correlation), title='{:.02}'.format(correlation))

g.barnes_hut(gravity=-1000, spring_length=20)
g.show_buttons()
g.show("output/topic_network_example.html")