import random
import pickle
import numpy as np
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as vs
from textstat.textstat import *
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
hate = open( "hate.txt", 'r' ).read()
offensive = open( 'offensive.txt', 'r' ).read()
none = open( 'nonenew.txt', 'r').read()



def converter(tweet):
    sent= cleaner.preprocessing_stage1(tweet)
    sent= " ".join(sent)
    final_tfidf= new_vectorizer.transform([sent]).toarray()
    final_tfidf= final_tfidf*idf_vals_

    sent1= cleaner.preprocessing_for_pos_tags([sent])
    final_pos= new_pos_vectorizer.transform(sent1).toarray()


    final_other= get_feature_array([tweet])

    transformed_tweet = np.concatenate([final_tfidf, final_pos, final_other],axis=1)

    
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
for i in hate.split('\n'):
    token_list= cleaner.preprocessing_stage1(i)
    document.append((token_list, 'Hate'))


for i in offensive.split('\n'):
    token_list= cleaner.preprocessing_stage1(i)
    document.append((token_list, 'Offensive'))


for i in none.split('\n'):
    token_list= cleaner.preprocessing_stage1(i)
    document.append((token_list, "None"))







features = []
targets = []
for f, t in document:                     
    features.append(" ".join(f))
    targets.append(t)


    
doc2 = []
for i in hate.split('\n'):
   doc2.append(i)
for i in offensive.split('\n'):
   doc2.append(i)
for i in none.split('\n'):
   doc2.append(i)




save_doc = open("doc.pickle", 'wb')
pickle.dump(document, save_doc)
save_doc.close()
print("Pickle of document is made.")


save_doc2 = open("doc2.pickle", "wb")
pickle.dump(doc2, save_doc2)
save_doc2.close()
print("Pickle of doc2 is made.")

# fx = open( "doc2.pickle", "rb" )
# doc2 = pickle.load( fx )
# fx.close()
#
# fx = open( "doc.pickle", "rb" )
# document = pickle.load( fx )
# fx.close()







target = []
for t in targets:
    if t == "Hate":
        target.append([0])
    elif t == "Offensive":
        target.append( [1] )
    else:
        target.append( [2] )

target = np.array(target)

save_new = open("new_pickle.pickle","wb")
pickle.dump(target, save_new)
save_new.close()
print("target has been pickled")

# fx = open("new_pickle.pickle", "rb")
# new = pickle.load(fx)
# fx.close()



vectorizer = TfidfVectorizer(
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

tfidf = vectorizer.fit_transform(features).toarray()
vocab = {v:i for i, v in enumerate(vectorizer.get_feature_names())}
idf_vals = vectorizer.idf_

feature_name_tfidf=[]
for k,v in vocab.items():
    feature_name_tfidf.append(k)


LR_tfidf=LogisticRegression(class_weight="balanced",penalty="l1",C=0.01).fit(tfidf,np.reshape(target, target.shape[0]))
select1=SelectFromModel(LR_tfidf,prefit=True)
x=select1.transform(tfidf)




save_tfidf = open("tfidf.pickle", "wb")
pickle.dump(tfidf, save_tfidf)
save_tfidf.close()
print("tfidf np array has been pickled")

# fx = open("tfidf.pickle", "rb")
# tfidf = pickle.load(fx)
# fx.close()





tweet_tags= cleaner.preprocessing_for_pos_tags(features)
save_tags = open("tags.pickle","wb")          
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




save_pos = open("pos.pickle", 'wb')
pickle.dump(pos, save_pos)
save_pos.close()
pprint("Pos_vectorizer has been pickled")




sentiment_analyzer = vs()


def more_feats(sent):
    text = cleaner.basic_cleaning( sent )
    sentiment = sentiment_analyzer.polarity_scores( sent )
    syllables = textstat.syllable_count( text )
    avg_syl_per_word = (0.001 + float( syllables )) / float( 0.001 + len( word_tokenize( text ) ) )
    num_terms = len( sent.split() )
    num_words = len( text.split() )
    num_unique_words = len( set( text.split() ) )
    num_char = len( text )
    total_char = len( sent )
    sent = cleaner.preprocessing_stage2( sent )
    urlcount = sent.count( "URLHERE" )
    mention = sent.count( "MENTIONHERE" )
    hashtags = sent.count( "HASHTAGHERE" )
    is_retweet = ("RT" in text)
    FKRA = round( float( 0.39 * float( num_words ) / 1.0 ) + float( 11.8 * avg_syl_per_word ) - 15.59, 1 )
    FRE = round( 206.835 - 1.015 * (float( num_words ) / 1.0) - (84.6 * float( avg_syl_per_word )), 2 )

    info_features = [FKRA, FRE, syllables, avg_syl_per_word, num_char, total_char, num_terms,
                     num_words, num_unique_words, sentiment['neg'], sentiment['pos'], sentiment['neu'],
                     sentiment['compound'],
                     hashtags, mention, urlcount, is_retweet]
    return info_features


def get_feature_array(doc):
    feats = []
    i = 0
    for t in doc:
        feats.append( more_feats( t ) )
        
    return np.array( feats )



extras = get_feature_array( doc2 )
extras_f = open( "extrafeats.pickle", "wb" )
pickle.dump( extras, extras_f )
extras_f.close()
print( "Extras has been pickled" )

# extras_f = open("extrafeats.pickle", 'rb')
# extras = pickle.load(extras_f)
# extras_f.close()



M = np.concatenate([x, y, extras, target], axis=1 )

print( "shape of M: ",M.shape )



np.random.shuffle(M )
y = np.reshape(M[:,-1],M.shape[0])


x_train= M[:int( M.shape[0] * 0.7 ),:-1]
x_test= M[int( M.shape[0] * 0.7 ):, :-1]

y_train= y[: int(y.shape[0]*0.7)]
y_test= y[int(y.shape[0]*0.7):]





RFclassifier = RandomForestClassifier( n_estimators=90, max_features=0.65, min_samples_leaf=12, bootstrap=True, n_jobs=2 )
RFclassifier.fit(x_train,y_train)
#pickle RFclassifier
save_rfc = open("RF_pickle.pickle", "wb")
pickle.dump(RFclassifier, save_rfc)
save_rfc.close()
print("RFclassifier has been pickled")

y_pred_rfc = RFclassifier.predict(x_test)
accuracyRF= find_accuracy(y_test, y_pred_rfc)
precisionRF= precision_recall_fscore_support(y_test, y_pred_rfc, average='weighted')[0]
print("Accuracy of Random forest is : ", accuracyRF)







svmclassifier = LinearSVC(class_weight='balanced',C=0.01, penalty='l2', loss='squared_hinge',multi_class='ovr')
svmclassifier.fit( x_train, y_train)
#pickle SVMclassifier
save_classifier = open("svm_pickle.pickle", "wb")
pickle.dump(svmclassifier, save_classifier)
save_classifier.close()

y_pred_svm = svmclassifier.predict(x_test)
accuracySVM= find_accuracy(y_test, y_pred_svm)
precisionSVM= precision_recall_fscore_support(y_test, y_pred_svm, average='weighted')[0]
print("Accuracy of SVM is : ", accuracySVM)




LRclassifier= LogisticRegression(class_weight='balanced',penalty="l2",C=0.01)
LRclassifier.fit(x_train, y_train)
#pickle LRclassifier
save_classifier = open("LR_pickle.pickle", "wb")
pickle.dump(LRclassifier, save_classifier)
save_classifier.close()


y_pred_lr = LRclassifier.predict(x_test)
accuracyLR= find_accuracy(y_test, y_pred_lr)
precisionLR= precision_recall_fscore_support(y_test, y_pred_lr, average='weighted')[0]
print("Accuracy of Logistic Regression is : ", accuracyLR)






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






