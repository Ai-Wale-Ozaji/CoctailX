
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
def load_data(path):
    df = pd.read_csv(path)
    # Trim accidental whitespace around column names
    df.columns = df.columns.str.strip()
    return df

DATA_PATH = "synthetic_cocktail_survey.csv"
df = load_data(DATA_PATH)

# ---------------- Helper utilities ------------------ #
def preprocess_binary(df_in: pd.DataFrame, target: str):
    X = df_in.drop(columns=[target])
    y = df_in[target].map({'Yes':1, 'No':0, 'Maybe':0})
    cat_cols = X.select_dtypes(include='object').columns
    num_cols = X.select_dtypes(exclude='object').columns
    pre = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ('num', StandardScaler(), num_cols)
    ])
    return pre, X, y

def confusion_fig(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['0','1'], yticklabels=['0','1'], ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    return fig

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))

def spend_to_num(x):
    return {'<₹100':50,'₹100–₹200':150,'₹200–₹300':250,'>₹300':350}.get(x, np.nan)

def persona_summary(d, label='Cluster'):
    return d.groupby(label).agg({
        'Age Group': lambda x: x.mode()[0],
        'Monthly Income': lambda x: x.mode()[0],
        'Preferred Packaging': lambda x: x.mode()[0],
        'Preferred Pack Size': lambda x: x.mode()[0],
        'Importance of Sustainability': lambda x: x.mode()[0]
    }).reset_index()

# --------------- Sidebar navigation ---------------- #
st.sidebar.title("🍸 Menu")
section = st.sidebar.radio("Go to", (
    "Data Visualisation", "Classification", "Clustering", "Association Rules", "Regression"))

# ============== Data Visualisation ================= #
if section == "Data Visualisation":
    st.header("📊 Data Visualisation")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Age Distribution")
        fig, ax = plt.subplots()
        sns.countplot(y='Age Group', data=df,
                      order=df['Age Group'].value_counts().index,
                      palette="Set3", ax=ax)
        st.pyplot(fig)
    with col2:
        st.subheader("Gender Split")
        fig, ax = plt.subplots()
        counts = df['Gender'].value_counts()
        ax.pie(counts, labels=counts.index, autopct="%1.0f%%", colors=PALETTE)
        ax.axis('equal')
        st.pyplot(fig)

# ============== Classification ===================== #
elif section == "Classification":
    st.header("🤖 Classification")
    possible_targets = [c for c in ['Buys on Bundle Deal','Willing to Switch Brand'] if c in df.columns]
    if not possible_targets:
        st.error("No suitable binary target columns found in data.")
        st.stop()
    target = st.selectbox("Target variable", possible_targets, index=0)
    pre, X, y = preprocess_binary(df, target)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
    models = {
        "KNN":KNeighborsClassifier(),
        "Decision Tree":DecisionTreeClassifier(random_state=42),
        "Random Forest":RandomForestClassifier(n_estimators=300, random_state=42),
        "Gradient Boosting":GradientBoostingClassifier(random_state=42)
    }
    metrics = []; fitted={}
    for name, mdl in models.items():
        pipe = Pipeline([('prep',pre),('model',mdl)])
        pipe.fit(X_train,y_train)
        preds = pipe.predict(X_test)
        metrics.append({
            "Model": name,
            "TrainAcc": pipe.score(X_train,y_train),
            "TestAcc": pipe.score(X_test,y_test),
            "Precision": precision_score(y_test, preds, zero_division=0),
            "Recall": recall_score(y_test, preds, zero_division=0),
            "F1": f1_score(y_test, preds, zero_division=0)
        })
        fitted[name]=pipe
    met_df = pd.DataFrame(metrics).round(2)
    st.dataframe(met_df, use_container_width=True)
    sel = st.selectbox("Confusion matrix for model", met_df['Model'])
    st.pyplot(confusion_fig(y_test, fitted[sel].predict(X_test)))
    st.subheader("ROC Curves")
    fig, ax = plt.subplots()
    for name, pipe in fitted.items():
        if hasattr(pipe.named_steps['model'],'predict_proba'):
            proba = pipe.predict_proba(X_test)[:,1]
            fpr,tpr,_ = roc_curve(y_test, proba)
            ax.plot(fpr,tpr,label=name)
    ax.plot([0,1],[0,1],'--',color='grey'); ax.legend(); st.pyplot(fig)

# ============== Clustering ========================= #
elif section == "Clustering":
    st.header("🌀 Customer Segmentation (K‑Means)")
    cat = df.select_dtypes(include='object').columns
    num = df.select_dtypes(exclude='object').columns
    pre = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown="ignore"), cat),
        ('num', 'passthrough', num)
    ])
    scaler = StandardScaler(with_mean=False)
    X_pre = pre.fit_transform(df)
    X_scaled = scaler.fit_transform(X_pre)
    inertias=[]; Ks=range(2,11)
    for k in Ks:
        inertias.append(KMeans(n_clusters=k, random_state=42, n_init='auto').fit(X_scaled).inertia_)
    fig, ax = plt.subplots()
    sns.lineplot(x=list(Ks),y=inertias,marker='o',ax=ax)
    ax.set_xlabel('Clusters k'); ax.set_ylabel('Inertia')
    st.pyplot(fig)
    k_val = st.slider("Pick k",2,10,3)
    km = KMeans(n_clusters=k_val, random_state=42, n_init='auto')
    labels = km.fit_predict(X_scaled)
    df_lab = df.copy(); df_lab['Cluster']=labels
    st.subheader("Persona summary")
    st.dataframe(persona_summary(df_lab))
    st.download_button("Download clusters", df_lab.to_csv(index=False),
                       file_name="clustered_data.csv", mime="text/csv")

# ============== Association Rules ================== #
elif section == "Association Rules":
    st.header("🔗 Apriori Rules")
    col_options = df.columns.tolist()
    chosen = st.multiselect("Columns for transactions", col_options,
                            default=[c for c in ['Preferred Alcohol Types','Interested Cocktail Flavours'] if c in df.columns])
    if not chosen:
        st.warning("Select at least one column.")
    else:
        sup = st.slider("Min support", 0.01,0.3,0.05,0.01)
        conf = st.slider("Min confidence",0.1,1.0,0.5,0.05)
        if st.button("Run Apriori"):
            transactions=[{item.strip() for col in chosen for item in str(row[col]).split(',')} for _,row in df[chosen].iterrows()]
            te=mlprep.TransactionEncoder()
            df_tf=pd.DataFrame(te.fit_transform(transactions), columns=te.columns_)
            freq=fp.apriori(df_tf, min_support=sup, use_colnames=True)
            if freq.empty:
                st.info("No frequent itemsets found with current support.")
            else:
                rules=fp.association_rules(freq, metric='confidence', min_threshold=conf)
                if rules.empty:
                    st.info("No rules satisfy confidence threshold.")
                else:
                    st.dataframe(rules.sort_values('confidence',ascending=False)
                                 .head(10)[['antecedents','consequents','support','confidence','lift']])

# ============== Regression ========================= #
else:
    st.header("📈 Regression Insights")
    df_reg = df.copy()
    df_reg['Spend'] = df_reg['Willingness to Pay (per unit)'].apply(spend_to_num)
    df_reg.dropna(subset=['Spend'], inplace=True)
    X_raw = df_reg.drop(columns=['Spend']); y = df_reg['Spend']
    cat = X_raw.select_dtypes(include='object').columns; num = X_raw.select_dtypes(exclude='object').columns
    pre = ColumnTransformer([('cat',OneHotEncoder(handle_unknown="ignore"),cat),
                             ('num',StandardScaler(),num)])
    X_train,X_test,y_train,y_test=train_test_split(X_raw,y,test_size=0.25,random_state=42)
    regs = {
        "Linear":LinearRegression(),
        "Ridge":Ridge(alpha=1.0),
        "Lasso":Lasso(alpha=0.01),
        "Decision Tree":DecisionTreeRegressor(random_state=42)
    }
    rows=[]; pipes={}
    for n,r in regs.items():
        p=Pipeline([('prep',pre),('model',r)])
        p.fit(X_train,y_train)
        preds=p.predict(X_test)
        rows.append({
            "Model":n,
            "MAE": np.mean(np.abs(y_test - preds)),
            "RMSE": rmse(y_test, preds),
            "R²": np.corrcoef(y_test, preds)[0,1]**2
        })
        pipes[n]=p
    res=pd.DataFrame(rows).round(1)
    st.dataframe(res, use_container_width=True)
    best=res.sort_values('RMSE').iloc[0]['Model']
    st.subheader(f"Actual vs Predicted – {best}")
    fig, ax = plt.subplots()
    ax.scatter(y_test, pipes[best].predict(X_test), color=PALETTE[0])
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    st.pyplot(fig)
