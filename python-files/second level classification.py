import random
import pickle
import numpy as np
from nltk.tokenize import word_tokenize
import preprocessing as cleaner
from statistics import mode
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support
from nltk.classify import ClassifierI
from sklearn.feature_selection import SelectFromModel


l_stop = 'a an and are as at be by for from has he in is it its of on the that to was were will with is are am'.split()

sexism = open("pdgsexism.txt", 'r',encoding="utf8").read()
sexism2 =  open("sexism.txt", 'r').read()
sexism3 = open('more-sexist.txt', 'r', encoding="utf8").read()
racism = open("racialtweets.txt", 'r').read()
racism2 = open("more-racial.txt", 'r', encoding="utf8").read()
racism3 = open('racial2.txt', 'r').read()
other = open('hateful-extra.txt', 'r', encoding="utf8").read()



def converter(tweet):
    sent= cleaner.preprocessing_stage1(tweet)
    sent= " ".join(sent)
    final_tfidf= new_vectorizer.transform([sent]).toarray()
    final_tfidf= final_tfidf*idf_vals_

    sent1= cleaner.preprocessing_for_pos_tags([sent])
    final_pos= new_pos_vectorizer.transform(sent1).toarray()

    transformed_tweet = np.concatenate([final_tfidf, final_pos],axis=1)

    
    return transformed_tweet



class voted_classifier(ClassifierI):
    

    def __init__(self,*classifiers):
        self._classifier=classifiers

    


    def predict_type(self,tweet):
        final_feature= converter(tweet)
        vote=[]
        for c in self._classifier:
            v=c.predict(final_feature)
            vote.append(int(v[0]))

        return mode(vote)
        
        
        
    
def find_accuracy(y_true, y_pred):
    count=0
    for i in range(len(y_true)):
        if y_pred[i] == y_true[i]:
            count += 1
    return count*100/len(y_true)


document = []
for i in sexism.split('\n'):
    token_list= cleaner.preprocessing_stage1(i)
    document.append((token_list, 'Sexism'))
for i in sexism2.split('\n'):
    token_list= cleaner.preprocessing_stage1(i)
    document.append((token_list, 'Sexism'))
for i in sexism3.split('\n'):
    token_list= cleaner.preprocessing_stage1(i)
    document.append((token_list, 'Sexism'))

for i in racism.split('\n'):
    token_list= cleaner.preprocessing_stage1(i)
    document.append((token_list, 'Racism'))
for i in racism2.split('\n'):
    token_list= cleaner.preprocessing_stage1(i)
    document.append((token_list, 'Racism'))
for i in racism3.split('\n'):
    token_list= cleaner.preprocessing_stage1(i)
    document.append((token_list, 'Racism'))

for i in other.split('\n'):
    token_list= cleaner.preprocessing_stage1(i)
    document.append((token_list, 'Other'))



#pickle document
save_doc = open("second_document.pickle", 'wb')
pickle.dump(document, save_doc)
save_doc.close()
print("Pickle of document is made.")




features = []
targets = []
for f, t in document:                     
    features.append(" ".join(f))
    targets.append(t)

    

#pickle features
save_feature = open("second_features.pickle", 'wb')
pickle.dump(features, save_feature)
save_feature.close()
print("Pickle of features is made.")


#pickle targets
save_target = open("second_targets.pickle", 'wb')
pickle.dump(targets, save_target)
save_target.close()
print("Pickle of targets is made.")






target = []
for t in targets:
    if t == "Sexism":
        target.append([0])
    elif t == "Racism":
        target.append( [1] )
    else:
        target.append( [2] )

target = np.array(target)





word_vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),
    stop_words=l_stop,
    use_idf=True,
    lowercase=True,
    smooth_idf=False,
    norm=None,
    decode_error='replace',
    max_features=10000,                          
    min_df=5,
    max_df=0.75
)


tfidf = word_vectorizer.fit_transform(features).toarray()
vocab = {v:i for i, v in enumerate(word_vectorizer.get_feature_names())}
idf_vals = word_vectorizer.idf_

feature_name_tfidf=[]
for k,v in vocab.items():
    feature_name_tfidf.append(k)


LR_tfidf=LogisticRegression(class_weight="balanced",penalty="l1",C=0.01).fit(tfidf,np.reshape(target, target.shape[0]))
select1=SelectFromModel(LR_tfidf,prefit=True)
x=select1.transform(tfidf)


#pickle tfidf
save_tfidf = open("second_tfidf.pickle", "wb")
pickle.dump(tfidf, save_tfidf)
save_tfidf.close()
print("tfidf np array has been pickled")





tweet_tags= cleaner.preprocessing_for_pos_tags(features)
#pickle tweet_tags
save_tags = open("second_tags.pickle","wb")          
pickle.dump(tweet_tags , save_tags)
save_tags.close()
print("pickle of tweet_tags is made")



pos_vectorizer = TfidfVectorizer(
    tokenizer=None,
    lowercase=False,
    preprocessor=None,
    ngram_range=(1, 3),
    stop_words=None,
    use_idf=False,
    smooth_idf=False,
    norm=None,
    decode_error='replace',
    max_features=5000,
    min_df=5,
    max_df=0.75,
)


pos = pos_vectorizer.fit_transform(tweet_tags).toarray()
pos_vocab = {v:i for i, v in enumerate(pos_vectorizer.get_feature_names())}
feature_name_pos=[]
for k,v in pos_vocab.items():
    feature_name_pos.append(k)


LR_pos=LogisticRegression(class_weight="balanced",penalty="l1",C=0.01).fit(pos,np.reshape(target, target.shape[0]))
select2=SelectFromModel(LR_pos,prefit=True)
y=select2.transform(pos)

#pickle pos
save_pos = open("second_pos.pickle", 'wb')
pickle.dump(pos, save_pos)
save_pos.close()
print("Pos_vectorizer has been pickled")





M = np.concatenate([x, y, target], axis=1 )
print( "shape of M: ",M.shape )

np.random.shuffle(M )
y = np.reshape(M[:,-1],M.shape[0])


x_train= M[:int( M.shape[0] * 0.8 ),:-1]
x_test= M[int( M.shape[0] * 0.8 ):, :-1]

y_train= y[: int(y.shape[0]*0.8)]
y_test= y[int(y.shape[0]*0.8):]





RFclassifier = RandomForestClassifier( n_estimators=90, max_features=0.65, min_samples_leaf=12, bootstrap=True, n_jobs=2 )
RFclassifier.fit(x_train,y_train)
#pickle RFclassifier
save_rfc = open("second_RF_pickle.pickle", "wb")
pickle.dump(RFclassifier, save_rfc)
save_rfc.close()
print("RFclassifier has been pickled")

y_pred_rfc = RFclassifier.predict(x_test)
print("Accuracy of Random forest is : ", find_accuracy(y_test, y_pred_rfc))







svmclassifier = LinearSVC(class_weight='balanced',C=0.01, penalty='l2', loss='squared_hinge',multi_class='ovr')
svmclassifier.fit( x_train, y_train)
#pickle SVMclassifier
save_classifier = open("second_svm_pickle.pickle", "wb")
pickle.dump(svmclassifier, save_classifier)
save_classifier.close()

y_pred_svm = svmclassifier.predict(x_test)
print("Accuracy of SVM is : ", find_accuracy(y_test, y_pred_svm))




LRclassifier= LogisticRegression(class_weight='balanced',penalty="l2",C=0.01)
LRclassifier.fit(x_train, y_train)
#pickle LRclassifier
save_classifier = open("second_LR_pickle.pickle", "wb")
pickle.dump(LRclassifier, save_classifier)
save_classifier.close()


y_pred_lr = LRclassifier.predict(x_test)
print("Accuracy of Logistic Regression is : ", find_accuracy(y_test, y_pred_lr))








ngram_indices=select1.get_support(indices=True)
ngram_features=[feature_name_tfidf[i] for i in ngram_indices]
new_vocab_tfidf={v:i for i,v in enumerate(ngram_features)}
idf_vals_ = []
for i in ngram_indices:
    idf_vals_.append(idf_vals[i])


new_vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),
    stop_words=l_stop, 
    use_idf=False,
    smooth_idf=False,
    norm=None, 
    decode_error='replace',
    min_df=1,
    max_df=1.0,
    vocabulary=new_vocab_tfidf
    )



tfidf_=new_vectorizer.fit(features)




pos_indices=select2.get_support(indices=True)
pos_features=[feature_name_pos[i] for i in pos_indices]
new_pos={v:i for i, v in enumerate(pos_features)}


new_pos_vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),
    stop_words=None, 
    use_idf=False,
    smooth_idf=False,
    norm=None, 
    decode_error='replace',
    min_df=1,
    max_df=1.0,
    vocabulary=new_pos
    )

pos_ = new_pos_vectorizer.fit(tweet_tags)




sent="give any sentence"
my_classifier= voted_classifier(RFclassifier,svmclassifier,LRclassifier)
ans= my_classifier.predict_type(sent)
print(ans)


