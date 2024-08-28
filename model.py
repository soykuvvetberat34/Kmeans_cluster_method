import numpy as np 
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

os.environ['LOKY_MAX_CPU_COUNT'] = '4'

veri=pd.read_csv("C:\\Users\\berat\\pythonEğitimleri\\python\\Makine Öğrenmesi\\Gözetimsiz_öğrenme\\veri.csv")
x=veri.iloc[:,3:]
kmeans=KMeans(n_clusters=3,init="k-means++",verbose=2)
kmeans.fit(x)
#oluşturulan kümelerin merkezlerini bulma
print(kmeans.cluster_centers_)


#elbow yöntemi ile cluster sayısı analizi
sonuclar=[]
for i in range(1,10):
    kmeans=KMeans(n_clusters=i,init="k-means++",random_state=42)
    kmeans.fit(x)
    sonuclar.append(kmeans.inertia_)

plt.plot(range(1,10),sonuclar)
plt.show()
    
