
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.cluster import KMeans
import mlxtend.frequent_patterns as fp
import mlxtend.preprocessing as mlprep

st.set_page_config(page_title="Cocktail Insights Dashboard", layout="wide", page_icon="🍹")

sns.set_theme(style="whitegrid", palette="Set2")
PALETTE = sns.color_palette("Set2")

@st.cache_data
def load_data(path:str):
    return pd.read_csv(path)

DATA_PATH = "synthetic_cocktail_survey.csv"
df = load_data(DATA_PATH)

# --------------------- Helpers ------------------- #
def preprocess_binary(df_in: pd.DataFrame, target: str):
    X = df_in.drop(columns=[target])
    y = df_in[target].map({'Yes':1,'No':0,'Maybe':0})
    cat = X.select_dtypes(include='object').columns.tolist()
    num = X.select_dtypes(exclude='object').columns.tolist()
    pre = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat),
        ('num', StandardScaler(), num)
    ])
    return pre, X, y

def plot_cm(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['0','1'], yticklabels=['0','1'], ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    return fig

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def persona_table(df_l, label='Cluster'):
    return df_l.groupby(label).agg({
        'Age Group': lambda x: x.mode()[0],
        'Monthly Income': lambda x: x.mode()[0],
        'Preferred Packaging': lambda x: x.mode()[0],
        'Preferred Pack Size': lambda x: x.mode()[0],
        'Importance of Sustainability': lambda x: x.mode()[0]
    }).reset_index()

def spend_to_num(x):
    mapping = {'<₹100':50,'₹100–₹200':150,'₹200–₹300':250,'>₹300':350}
    return mapping.get(x, np.nan)

# ------------------- UI ---------------------- #
st.sidebar.title("🍸 Menu")
section = st.sidebar.radio("Section", ("Data Visualisation","Classification","Clustering","Association Rules","Regression"))

# -------- Data Visualisation -------- #
if section == "Data Visualisation":
    st.header("📊 Exploratory Visuals")
    col1,col2 = st.columns(2)
    with col1:
        st.subheader("Age Distribution")
        fig, ax = plt.subplots()
        sns.countplot(y='Age Group', data=df, order=df['Age Group'].value_counts().index, ax=ax)
        st.pyplot(fig)
    with col2:
        st.subheader("Gender Split")
        fig, ax = plt.subplots()
        counts=df['Gender'].value_counts()
        ax.pie(counts, labels=counts.index, autopct='%1.0f%%', colors=PALETTE); ax.axis('equal')
        st.pyplot(fig)

# -------------- Classification -------------- #
elif section == "Classification":
    st.header("🤖 Classification")
    target = st.selectbox("Target variable", ['Buys on Bundle Deal','Willing to Switch Brand'], index=1)
    pre, X, y = preprocess_binary(df, target)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
    models = {
        'KNN':KNeighborsClassifier(),
        'Decision Tree':DecisionTreeClassifier(random_state=42),
        'Random Forest':RandomForestClassifier(n_estimators=300, random_state=42),
        'Gradient Boosting':GradientBoostingClassifier(random_state=42)
    }
    rows=[]; fitted={}
    for name, mdl in models.items():
        pipe=Pipeline([('pre',pre),('model',mdl)])
        pipe.fit(X_train,y_train)
        y_pred=pipe.predict(X_test)
        rows.append([name,
                     pipe.score(X_train,y_train),
                     pipe.score(X_test,y_test),
                     precision_score(y_test,y_pred,zero_division=0),
                     recall_score(y_test,y_pred,zero_division=0),
                     f1_score(y_test,y_pred,zero_division=0)])
        fitted[name]=pipe
    met = pd.DataFrame(rows, columns=['Model','TrainAcc','TestAcc','Precision','Recall','F1'])
    st.dataframe(met.style.format("{:.2f}"), use_container_width=True)
    cm_model = st.selectbox("Confusion matrix model", met['Model'])
    st.pyplot(plot_cm(y_test, fitted[cm_model].predict(X_test)))
    st.subheader("ROC Curves")
    fig, ax = plt.subplots()
    for name, pipe in fitted.items():
        if hasattr(pipe.named_steps['model'], 'predict_proba'):
            proba = pipe.predict_proba(X_test)[:,1]
            fpr,tpr,_ = roc_curve(y_test, proba)
            ax.plot(fpr,tpr,label=name)
    ax.plot([0,1],[0,1],'--',color='gray'); ax.legend(); st.pyplot(fig)

# --------------- Clustering --------------- #
elif section == "Clustering":
    st.header("🌀 K‑Means Clustering")
    # Build pipeline that outputs sparse One‑Hot then scale (with_mean=False)
    cat_cols = df.select_dtypes(include='object').columns
    num_cols = df.select_dtypes(exclude='object').columns
    pre = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', 'passthrough', num_cols)
    ])
    pipe = Pipeline([
        ('pre', pre),
        ('scale', StandardScaler(with_mean=False))
    ])
    X_scaled = pipe.fit_transform(df)
    # Elbow
    inertias=[]; Ks=range(2,11)
    for k in Ks:
        km=KMeans(n_clusters=k, random_state=42, n_init='auto').fit(X_scaled)
        inertias.append(km.inertia_)
    fig, ax = plt.subplots()
    sns.lineplot(x=list(Ks), y=inertias, marker='o', ax=ax)
    ax.set_xlabel('k'); ax.set_ylabel('Inertia')
    st.pyplot(fig)
    k_val = st.slider("Choose k",2,10,3)
    km_final = KMeans(n_clusters=k_val, random_state=42, n_init='auto')
    labels = km_final.fit_predict(X_scaled)
    df_lab = df.copy(); df_lab['Cluster'] = labels
    st.subheader("Cluster Personas")
    st.dataframe(persona_table(df_lab))
    st.download_button("Download labelled data", df_lab.to_csv(index=False), "clustered.csv","text/csv")

# --------------- Association Rules --------------- #
elif section == "Association Rules":
    st.header("🔗 Apriori Rules")
    default_cols = ['Preferred Alcohol Types','Interested Cocktail Flavours']
    chosen = st.multiselect("Columns", df.columns, default=default_cols)
    sup = st.slider("Min support",0.01,0.3,0.05,0.01)
    conf = st.slider("Min confidence",0.1,1.0,0.5,0.05)
    if st.button("Run"):
        transactions=[{i.strip() for col in chosen for i in str(row[col]).split(',')} for _,row in df[chosen].iterrows()]
        te=mlprep.TransactionEncoder()
        df_tf=pd.DataFrame(te.fit_transform(transactions), columns=te.columns_)
        freq=fp.apriori(df_tf, min_support=sup, use_colnames=True)
        rules=fp.association_rules(freq, metric='confidence', min_threshold=conf)
        st.dataframe(rules.sort_values('confidence', ascending=False).head(10)[['antecedents','consequents','support','confidence','lift']])

# ---------------- Regression ---------------- #
else:
    st.header("📈 Regression Insights")
    df_reg = df.copy(); df_reg['Spend'] = df_reg['Willingness to Pay (per unit)'].apply(spend_to_num)
    df_reg = df_reg.dropna(subset=['Spend'])
    X_raw = df_reg.drop(columns=['Spend']); y = df_reg['Spend']
    cat = X_raw.select_dtypes(include='object').columns; num = X_raw.select_dtypes(exclude='object').columns
    pre = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), cat),
                             ('num', StandardScaler(), num)])
    X_train,X_test,y_train,y_test = train_test_split(X_raw,y,test_size=0.25,random_state=42)
    regs = {'Linear':LinearRegression(),'Ridge':Ridge(alpha=1.0),'Lasso':Lasso(alpha=0.01),'Decision Tree':DecisionTreeRegressor(random_state=42)}
    rows=[]; pipes={}
    for n,r in regs.items():
        p=Pipeline([('pre',pre),('model',r)]); p.fit(X_train,y_train); yp=p.predict(X_test)
        rows.append([n, np.mean(np.abs(y_test-yp)), rmse(y_test, yp), np.corrcoef(y_test, yp)[0,1]**2])
        pipes[n]=p
    res=pd.DataFrame(rows, columns=['Model','MAE','RMSE','R²'])
    st.dataframe(res.style.format('{:.1f}'), use_container_width=True)
    best=res.sort_values('RMSE').iloc[0]['Model']
    st.subheader(f'Actual vs Predicted ({best})')
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=pipes[best].predict(X_test), ax=ax, color=PALETTE[0]); ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
    st.pyplot(fig)
