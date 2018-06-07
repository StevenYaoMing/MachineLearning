import os
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN,KMeans
from scipy.spatial.distance import pdist,squareform
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt

def getData():
    pathFile="G:\sampledata"
    fileName='complains.xlsx'
    fileToRead=os.path.join(pathFile,fileName)
    df=pd.read_excel(fileToRead,header=0,usecols=[0,3,4,15,16],names=['date','id','user','long','lat'])
    return df

def setToCluster():
    df=getData()
    print(df.head())
    
    df.dropna(subset=['long', 'lat'], how='any', inplace=True)
    
    df=df.loc[(df.long>90) & (df.long<115) & (df.lat>21) & (df.lat<30)].drop_duplicates()
    
    coords=df.as_matrix(columns=['long','lat'])
    
    distance_matrix=squareform(pdist(coords,(lambda x,y:haversine2(x,y))))
    
    db=DBSCAN(eps=600,min_samples=3,metric='precomputed')
    y_db=db.fit_predict(distance_matrix)

    #分组为-1的代表噪声点
    #
    df['cluster']=y_db

    #db=DBSCAN(eps=epsilon,min_samples=10,metric='haversine',algorithm='ball_tree').fit(np.radians(coords))
    #cluster_labels = db.labels_
    cluster=set(y_db)

    num_clusters = len(set(y_db))-(-1 in set(y_db))
    print('num of cluster is {} '.format(num_clusters))

    #去除掉异常点的类，-1
    df2=df.loc[df.cluster !=-1]
    
    clu_cente=[]

    for i in set(df2.cluster):

        kmCluster=df2.loc[(df2.cluster)==i,['long','lat']]

        km=KMeans(n_clusters=1)
        clf=km.fit(kmCluster)
        center=clf.cluster_centers_
        clu_cente.append([i,center[0][0],center[0][1]])

    center_df=pd.DataFrame(clu_cente,columns=['cluster','long_cent_clu','lat_cent_clu'])
    result=df2.merge(center_df,on='cluster',how='inner',)
    result.to_csv('G:\sampledata\outputs.csv',index=None)

    plt.scatter(x=df2['lat'],y=df2['long'],c=df2['cluster'])
    plt.show()

def haversine2(lonlat1, lonlat2):
    lon1,lat1 = lonlat1
    lon2,lat2 = lonlat2
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return int(c * r * 1000)

if __name__=='__main__':
    setToCluster()