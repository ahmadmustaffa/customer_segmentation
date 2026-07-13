import streamlit as st
import pickle
import numpy as np
# from train_model import model
# from train_model import cluster_df, pca_df_kmeans
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


st.header('Customer Segmentation')
st.write('This application segments customers based on their credit card usage and financial behavior. '
         'The segmentation was performed using the K-Means clustering algorithm, and a Decision Tree model is used '
         'to predict the cluster for new customers.')

st.subheader("Dataset Overview")

cluster_df = pd.read_csv("Clustered_Customer_Data.csv")
col1, col2, col3 = st.columns(3)

col1.metric("Customers", len(cluster_df))
col2.metric("Features", len(cluster_df.columns)-1)
col3.metric("Clusters", 3)

st.table(cluster_df.head())

# visualizing the clusters
pca_df_kmeans = pd.read_csv("pca_df_kmeans.csv")
fig = plt.figure(figsize=(8,8))
ax = sns.scatterplot(x="PCA1",y="PCA2",hue="cluster", data=pca_df_kmeans, palette=['red','green','blue','black','yellow','purple'])
plt.title("Clustering using K-Means Algorithm")
plt.show()
st.pyplot(fig)

# countplot
np.random.seed(42)
fig, ax = plt.subplots(figsize=(8, 6))
countplot_fig = sns.countplot(x='Cluster', data=cluster_df, ax=ax)
st.pyplot(fig)


# saving decision tree model
# filename = 'dectree_final_model.sav'
# pickle.dump(model, open(filename, 'wb'))
# st.write('decission tree model saved to disk')

# load the model from disk
loaded_model = pickle.load(open("dectree_final_model.sav", "rb"))
st.write('decission tree model loaded from disk')
# result = loaded_model.score(X_test, y_test)
# st.write(result,'% Acuuracy')

# prediction from user input
st.title('prediction')

with st.form("my_form"):
    balance = st.number_input(label='Balance range (1-19 500)',min_value=1, max_value=19500, value=1)
    balance_frequency = st.number_input(label='Balance Frequency range (0-1)',min_value=0, max_value=1, value=0)
    purchases = st.number_input(label='Purchases range (0-49 040)',min_value=0, max_value=49040, value=0)
    oneoff_purchases = st.number_input(label='OneOff Purchases range (0-40 761)',min_value=0, max_value=40761, value=0)
    installments_purchases = st.number_input(label='Installments Purchases range (0-22 500)',min_value=0, max_value=22500, value=0)
    cash_advance = st.number_input(label='Cash Advance range (0-47 137)',min_value=0, max_value=47137, value=0)
    purchases_frequency = st.number_input(label='Purchases Frequency range (0-1)',min_value=0, max_value=1, value=0)
    oneoff_purchases_frequency = st.number_input(label='OneOff Purchases Frequency range (0-1)',min_value=0, max_value=1, value=0)
    purchases_installment_frequency = st.number_input(label='Purchases Installments Frequency range (0-1)',min_value=0, max_value=1, value=0)
    cash_advance_frequency = st.number_input(label='Cash Advance Frequency range (0-1.5)',min_value=0.0, max_value=1.5, value=0.0)
    cash_advance_trx = st.number_input(label='Cash Advance Trx range (0-123)',min_value=0, max_value=123, value=0)
    purchases_trx = st.number_input(label='Purchases TRX range (0-358)',min_value=0, max_value=358, value=0)
    credit_limit = st.number_input(label='Credit Limit range (50-30 000)',min_value=50, max_value=30000, value=50)
    payments = st.number_input(label='Payments range (0-50 721)',min_value=0, max_value=50721, value=0)
    minimum_payments = st.number_input(label='Minimum Payments range (0-76 406)',min_value=0, max_value=76406, value=0)
    prc_full_payment = st.number_input(label='PRC Full Payment range (0-1)',min_value=0, max_value=1, value=0)
    tenure = st.number_input(label='Tenure range (6-12)',min_value=6, max_value=12, value=6)

    data = [[balance,balance_frequency,purchases,oneoff_purchases,installments_purchases,cash_advance,purchases_frequency,
             oneoff_purchases_frequency,purchases_installment_frequency,cash_advance_frequency,cash_advance_trx,purchases_trx,
             credit_limit,payments,minimum_payments,prc_full_payment,tenure]]

    submitted = st.form_submit_button("Submit")

if submitted:
    clust = loaded_model.predict(data)[0]
    st.write('Data Belongs to Cluster',clust)
