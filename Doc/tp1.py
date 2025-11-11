from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
from sklearn.preprocessing import add_dummy_feature
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

import numpy as np

x, y = make_regression(n_samples=100, n_features=1, noise=10, bias=30,random_state=42)
plt.scatter(x,y)
plt.show()
X=add_dummy_feature(x)
print(X.shape,y.shape)

w=np.linalg.inv(X.T @ X) @ X.T @ y 
plt.figure()
x_line=np.linspace(np.min(X),np.max(X),2)
y_line=w[0]+w[1]*x_line
plt.plot(x_line,y_line,color="red",label="droite de régression",linestyle="-")
plt.scatter(x,y)

plt.show()
#  descente de gradient
alpha=0.01
n_epoch=1000
w=[0,0] 
#Utilation de seed pour reproductibilite
#np.random.seed(42)
#w = np.random.randn(2, )  # Initialiser les poids avec des valeurs aléatoires

for epoch in range(n_epoch):
    w=w - 2*alpha/len(x) * X.T@(X@w-y)

x_line=np.linspace(np.min(X),np.max(X))
y_line=w[0]+w[1]*x_line
plt.plot(x_line,y_line,color="green",label="",linestyle="-")
plt.scatter(x,y)

plt.show() # de
x, y = make_regression(n_samples=100, n_features=1, noise=10, bias=30,random_state=42)

y=0.5*y**2+y+2
plt.scatter(x,y)
plt.show()

poly_f=PolynomialFeatures(2,include_bias=False)
x_poly=poly_f.fit_transform(x)
sgd_reg=LinearRegression() 
sgd_reg.fit(x_poly, y)

x_line=np.linspace(np.min(x),np.max(x),200).reshape(200,1)
x_line_poly=poly_f.fit_transform(x_line)
y_line=sgd_reg.predict(x_line_poly)

plt.plot(x_line,y_line,color="orange",linestyle="-")
plt.scatter(x,y)
plt.show()








