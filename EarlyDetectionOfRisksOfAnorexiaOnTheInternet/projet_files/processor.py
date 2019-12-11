
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def processTFIDF (Xtrain, Xtest):
    vectorizer = TfidfVectorizer(min_df=5, max_df = 0.8, sublinear_tf=True, use_idf=True,stop_words='english')
    train_corpus_tf_idf = vectorizer.fit_transform(Xtrain) 
    test_corpus_tf_idf = vectorizer.transform(Xtest)
    return train_corpus_tf_idf, test_corpus_tf_idf

def processLSA(train_corpus_tf_idf, test_corpus_tf_idf):
    svd = TruncatedSVD(100)
    lsa = make_pipeline(svd, Normalizer(copy=False))
    # Run SVD on the training data, then project the training data.
    train_corpus_lsa = lsa.fit_transform(train_corpus_tf_idf)
    test_corpus_lsa = lsa.transform(test_corpus_tf_idf)
    return train_corpus_lsa, test_corpus_lsa


def processLDA(train_corpus_tf_idf,test_corpus_tf_idf,Ytrain): 
    lda = LDA(n_components=100)  
    train_corpus_tf_idf_lda = lda.fit_transform(train_corpus_tf_idf.toarray(), Ytrain)  
    test_corpus_tf_idf_lda = lda.transform(test_corpus_tf_idf.toarray())
    return train_corpus_tf_idf_lda, test_corpus_tf_idf_lda



def ConstruireModeleDoc2Vec (tagged_data):
    max_epochs = 30
    vector_size = 100
    alpha = 0.025

    modelDoc2Vec = Doc2Vec(vector_size=vector_size,
                    alpha=alpha, 
                    min_alpha=0.00025,
                    min_count=1,
                    dm =1 , epochs = max_epochs)

    modelDoc2Vec.build_vocab(tagged_data)
    modelDoc2Vec.train(tagged_data,
                total_examples=modelDoc2Vec.corpus_count,
                epochs=modelDoc2Vec.epochs)
    return modelDoc2Vec 
    

def processDocToVec(Xtrain, Xtest):
    tagged_Xtrain = [TaggedDocument(words=word_tokenize(x.lower()), tags=[str(i)]) for i,x in enumerate (Xtrain)]
    tokenized_Xtest = [(word_tokenize(x.lower())) for x in (Xtest)]
    modelDoc2Vec = ConstruireModeleDoc2Vec(tagged_Xtrain)
    X_test = [modelDoc2Vec.infer_vector(x, steps=20) for x in tokenized_Xtest]
    X_train = [modelDoc2Vec.infer_vector(x.words, steps=20) for x in tagged_Xtrain]
    return X_train ,X_test



def processTfIdfLSATimeSeries(trainTextTimeSeries,testTextTimeSeries):
    
    trainLSATimeSeries=[]
    testLSATimeSeries =[]
    train=[]
    test=[]
    vectorizer = TfidfVectorizer(min_df=5, max_df = 0.8, sublinear_tf=True, use_idf=True,stop_words='english')
    svd = TruncatedSVD(100)
    lsa = make_pipeline(svd, Normalizer(copy=False))
    for i in range (len(trainTextTimeSeries[0])):
        Xtrain = trainTextTimeSeries[:,i] 
        train.append(lsa.fit_transform((vectorizer.fit_transform(Xtrain))))
        Xtest = testTextTimeSeries[:,i] 
        test.append(lsa.transform((vectorizer.transform(Xtest))))
    train = np.array(train)
    test = np.array(test)
    for i in range (len(trainTextTimeSeries)):
        trainLSATimeSeries.append(train[:,i])
    for i in range (len(testTextTimeSeries)):
        testLSATimeSeries.append(test[:,i])
    return (np.array(trainLSATimeSeries)),(np.array(testLSATimeSeries))

def processTfIdfLDATimeSeries(trainTextTimeSeries,testTextTimeSeries,Ytrain):
    
    trainLDATimeSeries=[]
    testLDATimeSeries =[]
    train=[]
    test=[]
    
    vectorizer = TfidfVectorizer(min_df=5, max_df = 0.8, sublinear_tf=True, use_idf=True,stop_words='english')
    lda = LDA(n_components=100)
    
    
    for i in range (len(trainTextTimeSeries[0])):
        Xtrain = trainTextTimeSeries[:,i] 
        train.append(lda.fit_transform((vectorizer.fit_transform(Xtrain).toarray()),Ytrain))
        Xtest = testTextTimeSeries[:,i] 
        test.append(lda.transform((vectorizer.transform(Xtest).toarray())))
    train = np.array(train)
    test = np.array(test)
    for i in range (len(trainTextTimeSeries)):
        trainLDATimeSeries.append(train[:,i])
    for i in range (len(testTextTimeSeries)):
        testLDATimeSeries.append(test[:,i])
    return (np.array(trainLDATimeSeries)),(np.array(testLDATimeSeries))

def processDocToVecTimeSeries(trainTextTimeSeries,testTextTimeSeries):
    
    trainDTVTimeSeries=[]
    testDTVTimeSeries =[]
    train=[]
    test=[] 
    
    for i in range (len(trainTextTimeSeries[0])):
        Xtrain = trainTextTimeSeries[:,i] 
        Xtest = testTextTimeSeries[:,i] 
    
        tagged_Xtrain = [TaggedDocument(words=word_tokenize(x.lower()), tags=[str(i)]) for i,x in enumerate (Xtrain)]
        tokenized_Xtest = [(word_tokenize(x.lower())) for x in (Xtest)]
        modelDoc2Vec = ConstruireModeleDoc2Vec(tagged_Xtrain)
        
        train.append([modelDoc2Vec.infer_vector(x.words, steps=20) for x in tagged_Xtrain])
        test.append([modelDoc2Vec.infer_vector(x, steps=20) for x in tokenized_Xtest])
        
    train = np.array(train)
    test = np.array(test)
    for i in range (len(trainTextTimeSeries)):
        trainDTVTimeSeries.append(train[:,i])
    for i in range (len(testTextTimeSeries)):
        testDTVTimeSeries.append(test[:,i])
    return (np.array(trainDTVTimeSeries)),(np.array(testDTVTimeSeries))