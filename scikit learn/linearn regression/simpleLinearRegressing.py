import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model



#getting data from csv file using pamdas
df=pd.read_csv('datasets/cleanedDataset/homeprices.csv')

print(df)
print(df.area)
print(df['area'])
print(df[['area','price']])
plt.scatter(df.area,df.price,c='r',marker='+')

area=np.array(df[['area']])
area.reshape(-1,1)
price=np.array(df[['price']])
price.reshape(-1,1)

reg = linear_model.LinearRegression()
print(dir(reg))
reg.fit(df[['area']],df.price)
area_to_predict =np.array(2000)
area_reshaped=area_to_predict.reshape(1,-1)
predicted_price=reg.predict(area) 
plt.plot(df.area,predicted_price)
plt.show()
