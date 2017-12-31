import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import axes3d



def autolabel(rects,ax):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.005*height,'%d' % int(height),ha='center', va='bottom')


def autolabel2(rects,ax,category):
    """
    Attach a text label above each bar displaying its height
    """
    i=0
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.005*height,'%s' % category[i],ha='center', va='bottom')
        i+=1




def compare_three_classifiers(accuracies,precisions):

    n= 3;
    ind= np.arange(n)
    width= 0.25

    fig, ax= plt.subplots()
    rects1= ax.bar(ind, accuracies,width,color='g')
    rects2= ax.bar(ind+width, precisions,width, color='b')

    ax.set_title('Comparison of Accuracy and Precision')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('RandomForest','SVM','LogisticRegression'))
    ax.legend((rects1[0], rects2[0]), ('Accuracy', 'Precision'))
    plt.show()
    
    
    
    

    



def confusion_matrix_voted_classifier(y_test,y_pred,list_of_labels):
    confusion_matrix = confusion_matrix(y_test,y_pred)
    matrix_proportions = np.zeros((3,3))
    for i in range(0,3):
        matrix_proportions[i,:] = confusion_matrix[i,:]/float(confusion_matrix[i,:].sum())

    names=list_of_labels
    confusion_df = pd.DataFrame(matrix_proportions, index=names,columns=names)
    plt.figure(figsize=(5,5))
    seaborn.heatmap(confusion_df,annot=True,annot_kws={"size": 12},cmap='gist_gray_r',cbar=False, square=True,fmt='.2f')
    plt.ylabel(r'True categories',fontsize=14)
    plt.xlabel(r'Predicted categories',fontsize=14)
    plt.tick_params(labelsize=12)

    plt.show()





def sample_ploting(sum_tfidf,num_words,FKRA,target,list_labels):
    x1=[]
    y1=[]
    z1=[]
    x2=[]
    y2=[]
    z2=[]
    x3=[]
    y3=[]
    z3=[]

    for i in range(len(target)):
        if target[i]== 0:
            x1.append(sum_tfidf[i])
            y1.append(num_words[i])
            z1.append(FKRA[i])

        elif target[i]== 1:
            x2.append(sum_tfidf[i])
            y2.append(num_words[i])
            z2.append(FKRA[i])

        else:
            x3.append(sum_tfidf[i])
            y3.append(num_words[i])
            z3.append(FKRA[i])


    fig= plt.figure()
    ax1=fig.add_subplot(111,projection='3d')

    ax1.scatter(x1,y1,z1,c='r',s=50,label=list_labels[0])
    ax1.scatter(x2,y2,z2,c='g',s=60, marker="*",label=list_labels[1])
    ax1.scatter(x3,y3,z3,c='b',s=50, marker="^",label=list_labels[2])

    ax1.set_xlabel('sum of tfidf of words')
    ax1.set_ylabel('number of words')
    ax1.set_zlabel('FKRA of sentence')
    ax1.legend()
    plt.show()




def corpus_distribution(num_of_each_labels,tuple_of_label_name):

    n= 3;
    ind= np.arange(n)
    width= 0.25

    fig, ax= plt.subplots()
    rects1= ax.bar(ind,num_of_each_labels ,width,color='g')
    

    ax.set_title('Corpus Distribution')
    ax.set_xticks(ind)
    ax.set_xticklabels(tuple_of_label_name)
    autolabel(rects1,ax)
    plt.show()





def show_predictions(confidence,category):
    n= len(confidence);
    ind= np.arange(n)
    width= 0.25
    l=[]
    for i in range(len(confidence)):
        l.append(i+1)
        
    fig, ax= plt.subplots()
    rects1= ax.bar(ind,confidence,width,color='r')
    ax.set_title('Prediction')
    ax.set_xticks(ind)
    ax.set_xticklabels(tuple(l))
    autolabel2(rects1,ax,category)
    plt.show()
    













target=[0,1,2,1]
sum_tfidf=[3.4,5.6,4.2,4.7]
num_words=[5,6,7,5]
FKRA=[3.4,6.7,8.4,9.5]
list_labels=['Hate','Offensive','None']

    

sample_ploting(sum_tfidf,num_words,FKRA,target,list_labels)


num_of_each_labels=[23,45,67]
tuple_of_label_name=('A','B','C')

corpus_distribution(num_of_each_labels,tuple_of_label_name)



accuracies=[85.3,87.4,86.3]
precisions=[76,84,94]

compare_three_classifiers(accuracies,precisions)




confidence=[90,85,87,89]
category=["offensive","hate","none","none"]
show_predictions(confidence,category)






    
