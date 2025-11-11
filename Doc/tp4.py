import numpy as np

# import des outils d'affichage
import matplotlib.pyplot as plt

# import des outils de machine learning
import sklearn as skl
import sklearn.datasets as data
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.preprocessing import add_dummy_feature
from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import mglearn

x,y=make_classification(n_samples=1000,n_features=2,n_informative=2,n_redundant=0,n_clusters_per_class=1,random_state=26)

def plot_data(x, y):
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Data Visualization')
    plt.show()
plot_data(x, y)

X=add_dummy_feature(x)
clf = Perceptron(tol=1e-3,max_iter=1000, random_state=0)
clf.fit(X, y)

w1=clf.coef_[0].copy()
#Desente de gradient version batch logistique regression avec la fonction sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

w=np.zeros(X.shape[1]) # features = donn
n=X.shape[0] #
print(w,"initial weights")
def batch_logistic_regression_gradient_descent(X, y, alpha=1, w=0,n=0, n_iterations=10):
    for epoch in range(n_iterations):
        for j in range(n):
            xi = X[j]            # vecteur (1, x1, x2)
            yi = y[j]            # 0 ou 1
            z = np.dot(w, xi)
            pred = sigmoid(z)
            grad = xi * (pred - yi)   # vecteur gradient pour cet exemple
            w = w - alpha * grad      # mise à jour : w^{t+1} = w^t - alpha * xi * (sigma(w^T xi) - yi)
    return w
w=batch_logistic_regression_gradient_descent(X, y, alpha=1,w=w,n=n, n_iterations=10)
w2=w.copy()
print("w2 (batch logistic SGD) =", w2)

def plot_batch_frontiere(w,X,label,color):
    if X.shape[1] == w.shape[0]: # cas avec biais dans X
        X_plot = X[:, 1:3]
    else:
        X_plot = X[:, :2]
    x_min, x_max = X_plot[:,0].min() - 1, X_plot[:,0].max() + 1

    x1 = np.linspace(x_min, x_max, 200)
    if abs(w[2]) > 1e-8: # cas non dégénéré
        # ligne x2 = -(w0 + w1*x1)/w2
        x2 = -(w[0] + w[1]*x1) / w[2]
        plt.plot(x1, x2, color=color, label=label)
    else:
        # cas dégénéré (vertical)
        x_vert = -w[0] / (w[1] + 1e-12)
        plt.axvline(x=x_vert, color=color, label=label)
   
def plot_batch_frontiereVariant(w,X,label,color): # a mettre dans la documentation aussi
    x_min, x_max = X[:,1].min(), X[:,1].max()
    x1 = np.linspace(x_min, x_max, 100)
    x2 = -(w[0] + w[1]*x1)/w[2]
    plt.plot(x1, x2, label=label)
   




    # Tracé des points
plt.figure()
plt.scatter(X[:, 1], X[:, 2], c=y, edgecolor='k')
# Tracé des frontières de décision
plot_batch_frontiere(w1, X, 'perceptron', 'b')
plot_batch_frontiere(w2, X, 'batch logistic regression', 'g')
plt.legend()
plt.show()

from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=200, noise=0.25, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

mlp = MLPClassifier(hidden_layer_sizes=(10,), solver='lbfgs',
                    activation='relu', max_iter=500, random_state=0)
mlp.fit(X_train, y_train)

print("Score d'entraînement :", mlp.score(X_train, y_train))
print("Score de test :", mlp.score(X_test, y_test))

for n in range(10,100,10):
    mlp = MLPClassifier(hidden_layer_sizes=(n, n), solver='lbfgs',
                        activation='relu', max_iter=500, random_state=0)
    mlp.fit(X_train, y_train)
    print(f"{n} neurones × 2 couches — Test score: {mlp.score(X_test, y_test):.3f}")


