from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import mglearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,ConfusionMatrixDisplay
import pandas as pd
from sklearn.preprocessing import LabelEncoder


x,y=make_classification(n_samples=1000,n_features=2,n_informative=2,n_redundant=0,n_clusters_per_class=1,random_state=26)

def plot_data(x,y):
    print(y)
    plt.scatter(x[:,0],x[:,1],c=y)
    plt.show()
plot_data(x,y)


model=LogisticRegression()
model.fit(x,y)

#print(model.predict(x))
prediction=model.predict(x)

print(classification_report(y,prediction))


def plot_decision(x,model):
    plt.scatter(x[:,0],x[:,1],c=model.predict(x))
    w0=model.intercept_[0]
    w1,w2=model.coef_.T

    c1= -w0/w2
    c2=-w1/w2

    x1min,x1max=x[:,0].min(),x[:,0].max()

    xd= np.array([x1min, x1max])


    y=c1+c2*xd
    plt.plot(xd,y,color="red")
    plt.show()

plot_decision(x,model)
print("t")

clf = GaussianNB()
clf.fit(x,y)
prediction=clf.predict(x)
print(classification_report(y,prediction))

def plot_frontiere(x,model): #fonction pour tracer la frontiere de decision pour GaussianNB
    plt.scatter(x[:,0],x[:,1],c=model.predict(x))
    x1min,x1max=x[:,0].min(),x[:,0].max()
    x2min,x2max=x[:,1].min(),x[:,1].max()
# meshgrid cree une grille de points entre les min et max des deux dimensions
    xx1,xx2=np.meshgrid(np.linspace(x1min,x1max,100),np.linspace(x2min,x2max,100)) #creer une grille de points
    #ravel aplati la matrice en un vecteur
    Z=model.predict(np.c_[xx1.ravel(),xx2.ravel()]).reshape(xx1.shape) #predire la classe pour chaque point de la grille
    plt.contourf(xx1,xx2,Z,alpha=0.3)
    plt.show()
plot_frontiere(x,clf)


#KNN Intro pipline
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=26)
knn_pipline=make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=5))
knn_pipline.fit(x_train,y_train)
y_pred=knn_pipline.predict(x_test)
print(classification_report(y_test,y_pred))
cm=confusion_matrix(y_test,y_pred)
disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=knn_pipline.classes_)
disp.plot()
plt.show()
mglearn.plots.plot_knn_classification(n_neighbors=3)  #Test
plt.show()

    

# 1️⃣ Chargement du jeu de données
data = pd.read_csv("loan_data.csv")

# 2️⃣ Encodage de la colonne 'purpose' (catégorielle → numérique)
label_encoder = LabelEncoder()
data["purpose"] = label_encoder.fit_transform(data["purpose"])

# 3️⃣ Séparation X / y
X = data.drop("not.fully.paid", axis=1)
y = data["not.fully.paid"]

# 4️⃣ Division en train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 5️⃣ Normalisation des données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6️⃣ Entraînement et évaluation des modèles

# --- Régression logistique ---
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
y_pred_log = log_reg.predict(X_test_scaled)
print("Régression logistique :")
print(classification_report(y_test, y_pred_log))

# --- Bayésien naïf ---
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
y_pred_nb = nb.predict(X_test_scaled)
print("Bayésien naïf :")
print(classification_report(y_test, y_pred_nb))

# --- k-plus proches voisins ---
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
print("k-NN :")
print(classification_report(y_test, y_pred_knn))