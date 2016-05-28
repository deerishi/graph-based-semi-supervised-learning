from sklearn.datasets import fetch_20newsgroups
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import numpy as np
from nltk import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import os
import re
from os.path import expanduser
from sklearn.metrics.pairwise import pairwise_distances
                                                                                                                                                                                                                                                                                                 
from copy import copy

categories=['alt.atheism', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'),categories=categories)

class StemmerTokenizer(object):

    def __init__(self): 
        self.stemmer = PorterStemmer()

    def __call__(self, doc):
        return [self.stemmer.stem(t) for t in word_tokenize(doc)]
        

stop=stopwords.words('english')
stemmer=PorterStemmer()
lemmer=WordNetLemmatizer()
home=expanduser("~")

nonWords=['[',']','{','}','/','_','|']
    
def preProcess(dataset,target):
    processed=[]
    target2=target.tolist()
    labels=[]
    counter=-1
    j=1
    for line in dataset:
        #words=line.split(' ')
        
        print ' at j = ',j,'/',len(dataset)
        j=j+1
        counter+=1
        line=line.lower()
        line=line.strip()
        if len(line)>0:
            labels.append(target2[counter])
        else:
            continue
                   
        cleaned1=re.sub(r'[,.!?&$@%/\\"-;:()+*^`<>#`~=]',"",line)
        cleaned2=re.sub(r'\d',"",cleaned1)
        cleaned2=re.sub(r"'s","",cleaned2)
        cleaned2=re.sub(r"'","",cleaned2)
        cleaned2=re.sub(r'"',"",cleaned2)
        
        for ch in nonWords:
            try:
                cleaned2=cleaned2.replace(ch,'')
            except:
                pass
        words=nltk.word_tokenize(cleaned2)
        #print 'words is ',words
        tags=nltk.pos_tag(words)
        sentence=""
        sentence+=words[0]+" "
        for i in range(1,len(tags)):
            tag=tags[i]
            #print 'tag is ',tag
            try:
                if tag[1][0].lower() in ['a','v','n','s','r']:
                    word=lemmer.lemmatize(tag[0],tag[1][0].lower())
                else:
                    word=lemmer.lemmatize(tag[0])
                sentence+=word+" "
            except Exception:
                pass
            
            
        #print 'the lemmatized sentence is ',sentence
        for stopWord in stop:
            try:
                cleaned2=sentence.replace(stopWord,"")
            except Exception:
                pass
            
        cleaned2.strip()
        processed.append(cleaned2)

    return processed,labels
        
smallData=newsgroups_train.data
target=newsgroups_train.target
j=0
'''
numwords=0
for data in smallData:
    print 'data j=',j,' is ',data,'\n\n'
    j+=1'''    
processedData,labels=preProcess(smallData,target)
parray=np.asarray(processedData)
np.save('FullData',parray)
labels=np.asarray(labels)
np.save('Labels',labels)

i=0
words={}
for line in processedData:
    #print 'i= ',i,' ',line,'\n\n'
    ws=(line.split(' '))
    for w in ws:
        words[w]=1
    #print 'original sentence i= ',i,' is ',newsgroups_train.data[i],'\n\n'
    i+=1

vectorizer = TfidfVectorizer(decode_error='replace',analyzer='word',stop_words='english',lowercase=True,tokenizer=StemmerTokenizer())
vectorizer.fit(processedData)
vectors = vectorizer.transform(processedData)
print 'vectorizer is  ' ,vectors.shape
print 'numwords is ',len(words)
print 'get_feature_names is ',vectorizer.get_feature_names()
pwdis=pairwise_distances(vectors,metric='cosine')

print 'pairwise_distances is ',pwdis.shape,'\n'
print 'labels .shape is ',len(labels)
