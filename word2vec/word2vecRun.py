import time
from gensim.models import word2vec, KeyedVectors
from gensim.models import Word2Vec
import multiprocessing
from gensim.models.word2vec import LineSentence
import logging


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # Word2Vec第一个参数代表要训练的语料
    # sg=1 表示使用Skip-Gram模型进行训练
    # size 表示特征向量的维度，默认为100。大的size需要更多的训练数据,但是效果会更好. 推荐值为几十到几百。
    # window 表示当前词与预测词在一个句子中的最大距离是多少
    # min_count 可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5
    # workers 表示训练的并行数
    #sample: 高频词汇的随机降采样的配置阈值，默认为1e-3，范围是(0,1e-5)

def A():
    #首先打开需要训练的文本
    shuju = open('/word2vec/JourneytotheWest.txt', 'rb')
    #通过Word2vec进行训练
    model = Word2Vec(LineSentence(shuju), sg=1,size=100, window=10, min_count=5, workers=15,sample=1e-3)
    print(model)

    #保存训练好的模型
    model.save('G:\pythonProject\AlgorithmProject\ThreeCountryPlay.word2vec')
    model.save('word2vec.model')
    print('训练完成')

if __name__ == '__main__':
    A()

#调用 加载
t1 = time.time()
model = Word2Vec.load('ThreeCountryPlay.word2vec')
t2 = time.time()

print(model)
print('.model load time %.4f'%(t2 - t1))

# model.init_sims(replace=True)
vectors = model.wv.vectors
words = model.wv.index2word

vec = model.wv['张飞']
print("张飞的词向量为：")
print(vec)

'''
几个相似性度量API
'''

# Compute the Word Mover's Distance between two documents.
# 计算两个文档的相似度——词移距离算法
# model.wv.wmdistance()

# Compute cosine similarity between two sets of words.
# 计算两列单词之间的余弦相似度——也可以用来评估文本之间的相似度
# model.wv.n_similarity(ws1, ws2)

# Compute cosine similarities between one vector and a set of other vectors.
# 计算向量之间的余弦相似度
# model.wv.cosine_similarities(vector_1, vectors_all)

# Compute cosine similarity between two words.
# 计算2个词之间的余弦相似度
# model.wv.similarity(w1, w2)

# Find the top-N most similar words.
# 找出前N个最相似的词
# model.wv.most_similar(positive=None, negative=None, topn=10, restrict_vocab=None, indexer=None)

# sent1 = ['张飞', '新能源', '运营', '航天', '新能源汽车', '平台', '城市', '打造', '技术', '携手']
# sent2 = ['新能源', '奇瑞', '新能源汽车', '致力于', '支柱产业', '整车', '汽车', '打造', '产业化', '产业基地']
# sent3 = ['辉瑞', '阿里', '互联网', '医师', '培训', '公益', '制药', '项目', '落地', '健康']
# sent4 = ['互联网', '医院', '落地', '阿里', '健康', '就医', '流程', '在线', '支付宝', '加速']

wd1 = ['玄德']
wd2 = ['云长']
wd3 = ['献帝']

wordsim1 = model.wv.n_similarity(wd1, wd2)
wordsim2 = model.wv.n_similarity(wd1, wd3)

print('玄德和云长相似性为：' + str(wordsim1))
print('玄德和献帝相似性为：' + str(wordsim2))
