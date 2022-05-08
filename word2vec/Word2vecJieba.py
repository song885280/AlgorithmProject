
from gensim.models.word2vec import Word2Vec
import jieba.analyse
import codecs
f = codecs.open('/word2vec/ThreeCountryPlay.txt', 'r', encoding ="utf-8")
target = codecs.open('/word2vec/JourneytotheWest.txt', 'w', encoding ="utf-8")

print('open files')
line_num = 1
line = f.readline()

while line:
    print('----- processing ', line_num, 'article---------------')
    line_seg = ' '.join(jieba.cut(line))
    target.writelines(line_seg)
    line_num = line_num + 1
    line = f.readline()

f.close()
target.close()
exit()
