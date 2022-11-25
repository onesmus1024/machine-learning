from statistics import mode
import numpy as np
import matplotlib.pyplot as plt
from  sklearn.linear_model import LinearRegression


x = np.arange(10)
y = 4*x + 2


y = y.reshape(-1,1)
x= x.reshape(-1,1)
print(x.shape)
print(y.shape)
# print(y)
# print(x)

# visualization
# plt.plot(x,y)
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend("test a simple linear regression")
# plt.show()


model = LinearRegression()
model.fit(x,y)
print(model.predict([[8]]))