import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC



X,y=make_blobs(n_samples=300,centers=4,random_state=42) # make_blobs pour generer des donnees de classification

def plot_data(X,y):
    plt.scatter(X[:,0],X[:,1],c=y)
    plt.show()

plot_data(X,y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42) # diviser les donnees en train et test
#train c'est pour entrainer le modele
#test c'est pour evaluer le modele
tree=DecisionTreeClassifier(max_depth=10,random_state=42) # creer un modele d'arbre de decision
tree.fit(X_train,y_train) # entrainer le modele
print(classification_report(y_test, tree.predict(X_test)))
print("Accuracy:", accuracy_score(y_test, tree.predict(X_test))) # evaluer le modele
def plot_decision_tree(tree):
     plt.figure(figsize=(12,8)) # 12 pouces de large et 8 pouces de haut
     plot_tree(tree, filled=True, feature_names=['x1','x2'] )
     plt.show()
plot_decision_tree(tree)
#Variante avec  graphviz

#from sklearn import tree as sk_tree
#import graphviz
#from IPython.display import display
#graphviz_tree = sk_tree.export_graphviz(tree)
#display(graphviz.Source(graphviz_tree))
#
def plot_contours(x,y,tree):   
    lengths, widths = np.meshgrid(np.linspace(np.min(x[:,0]),np.max(x[:,0]), 100), np.linspace(np.min(x[:,1]),np.max(x[:,1]), 100))
    all = np.c_[lengths.ravel(), widths.ravel()]
    y_pred = tree.predict(all).reshape(lengths.shape)
    plt.contourf(lengths, widths, y_pred,alpha=0.3)
    res=plt.scatter(x[:,0],x[:,1],marker='o', c=y)
    plt.legend(*res.legend_elements(),loc="upper right", title="Classes") # *res.legend_elements() pour recuperer les elements de la legende

plot_contours(X, y, tree)
plt.show()


svm = SVC(kernel='linear',C=1.0,gamma=0.05) # C 
svm.fit(X_train, y_train)
def plot_svm_contours(x,y,svm):   
    lengths, widths = np.meshgrid(np.linspace(np.min(x[:,0]),np.max(x[:,0]), 100), np.linspace(np.min(x[:,1]),np.max(x[:,1]), 100))
    all = np.c_[lengths.ravel(), widths.ravel()]
    y_pred = svm.predict(all).reshape(lengths.shape)
    plt.contourf(lengths, widths, y_pred,alpha=0.3)
    res=plt.scatter(x[:,0],x[:,1],marker='o', c=y)
    plt.legend(*res.legend_elements(),loc="upper right", title="Classes") # *res.legend_elements() pour recuperer les elements de la legende
    plt.show()

plot_svm_contours(X, y, svm)



