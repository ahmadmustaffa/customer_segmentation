

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans #,AgglomerativeClustering,DBSCAN,SpectralClustering
# from sklearn.mixture import GaussianMixture
# from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import io

from sklearn import metrics


# st.write('this is running')
df = pd.read_csv('Customer Data.csv')
# st.table(df.head(10))


# shape and info of dataframe
# st.write("Shape of the DataFrame:", df.shape)
# st.write("Info of the DataFrame:")
buffer = io.StringIO()
df.info(buf=buffer)
info = buffer.getvalue()
# st.text(info)

# st.write("Describe DataFrame:")
# st.table(df.describe())


# st.write("Null values in the DataFrame:")
# st.table(df.isnull().sum())
# imputing mean in null values
df.dropna(subset=['CREDIT_LIMIT'], inplace=True)
mean = df['MINIMUM_PAYMENTS'].mean()
df['MINIMUM_PAYMENTS'].fillna(mean, inplace=True)

# st.write("Null values after cleaning:")
# st.table(df.isnull().sum())
# st.write("Number of duplicates:", df.duplicated().sum())

df.drop(columns=["CUST_ID"],axis=1,inplace=True)
# st.write("CUST_ID column deleted")


# heatmap
fig = plt.figure(figsize=(10, 8)) 
sns.heatmap(df.corr(), annot=True)
# st.pyplot(fig)


# scaling dataframe
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)
# st.write("dataframe is scaled")


# perform PCA with different numbers of components
n_components = range(1, len(df.columns) + 1)
explained_variances = []

for n in n_components:
    pca = PCA(n_components=n)
    pca.fit(scaled_df)
    explained_variances.append(pca.explained_variance_ratio_.sum())


# plot the explained variance ratio
plt.figure()
plt.plot(n_components, explained_variances, 'bo-')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Elbow Method for Optimal Number of Components')
# plt.savefig('elbow_plot for n_components.png')
# st.write('plot saved for n_components')
# st.pyplot(plt)


# doing pca
# st.write('n components in pca is 6')
pca = PCA(n_components=6)
principal_components = pca.fit_transform(scaled_df)
pca_df = pd.DataFrame(data=principal_components ,columns=["PCA1","PCA2","PCA3","PCA4","PCA5","PCA6"])
# st.table(pca_df.head(10))


# finding value of k in kmeans
inertias = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_df)
    inertias.append(kmeans.inertia_)


# plot the elbow curve
plt.figure()
plt.plot(k_values, inertias, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Value of K')
# plt.savefig('elbow_plot for k kmeans.png')
# st.write('plot saved for k kmeans')
# st.pyplot(plt) #taking time to plot on web


#kmeans
# st.write('selecting k value 3')
kmeans_model=KMeans(3)
kmeans_model.fit_predict(scaled_df)
pca_df_kmeans= pd.concat([pca_df,pd.DataFrame({'cluster':kmeans_model.labels_})],axis=1)
pca_df_kmeans.to_csv("pca_df_kmeans.csv", index=False)


# visualizing the clusters
fig = plt.figure(figsize=(8,8))
ax = sns.scatterplot(x="PCA1",y="PCA2",hue="cluster", data=pca_df_kmeans, palette="Set2")
plt.title("Clustering using K-Means Algorithm")
# plt.show()
# st.pyplot(fig)


# find all cluster centers
cluster_centers = pd.DataFrame(data=kmeans_model.cluster_centers_,columns=[df.columns])
# inverse transform the data
cluster_centers = scaler.inverse_transform(cluster_centers)
cluster_centers = pd.DataFrame(data=cluster_centers,columns=[df.columns])
# st.table(cluster_centers)


# dataframe with cluster segement
cluster_df = pd.concat([df,pd.DataFrame({'Cluster':kmeans_model.labels_})],axis=1)
# st.table(cluster_df.head(15))


# cleaning null values in cluster dataframe
# st.write("Null values in cluster DataFrame:")
# st.table(cluster_df.isnull().sum())
# st.write('null rows in cluster DataFrame')
# st.table(cluster_df[cluster_df.isnull().any(axis=1)])
cluster_df.dropna(inplace=True)
# st.write('null values in cluster DataFrame removed')
# st.write("after removing Null values in cluster DataFrame:")
# st.table(cluster_df.isnull().sum())


# visualizing the clusters
np.random.seed(42)
fig, ax = plt.subplots(figsize=(8, 6))
countplot_fig = sns.countplot(x='Cluster', data=cluster_df, ax=ax)
# st.pyplot(fig)


# saving kmeans_model
# joblib.dump(kmeans_model, "kmeans_model.pkl")
# st.write('kmeans model saved')


#saving clustered dataframe
cluster_df.to_csv("Clustered_Customer_Data.csv")
# st.write('clustered dataframe saved')


# split dataset in training and test sets
X = cluster_df.drop(['Cluster'],axis=1)
y= cluster_df['Cluster']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# st.write('dataset splitted')


# train and predict the model
model= DecisionTreeClassifier(criterion = 'entropy')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# st.write('kmeans model trained and predicted')


# confusion matrix and classification report
# st.write('confusion matrix and classification report')
# st.table(confusion_matrix(y_test, y_pred))
# st.table(classification_report(y_test, y_pred))


# confusion matrix
cm = metrics.confusion_matrix(y_test, y_pred)
# st.write("Confusion Matrix")
# st.dataframe(pd.DataFrame(cm))

# classification report
cr = metrics.classification_report(y_test, y_pred, output_dict=True)
# st.write("Classification Report")
# st.dataframe(pd.DataFrame(cr).transpose())


# saving decision tree model

 
# after some time ...
 
# load the model from disk



# prediction from user input
