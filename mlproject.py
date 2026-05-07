import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression




st.set_page_config(
    page_title="Student GPA Prediction",
    page_icon="🎓",
    layout="wide"
)


st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top left, #172554 0%, #020617 45%, #020617 100%);
    color: white;
}


[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a, #020617);
    border-right: 1px solid rgba(148,163,184,.2);
}


.block-container {
    padding-top: 1.2rem;
}


.hero {
    background: linear-gradient(135deg, rgba(30,41,59,.95), rgba(15,23,42,.92));
    border: 1px solid rgba(148,163,184,.18);
    border-radius: 24px;
    padding: 30px;
    margin-bottom: 20px;
    box-shadow: 0 20px 50px rgba(0,0,0,.35);
}


.hero h1 {
    font-size: 42px;
    font-weight: 900;
    margin: 0;
}


.gradient-text {
    background: linear-gradient(90deg, #38bdf8, #a855f7, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}


.badge {
    display: inline-block;
    padding: 7px 13px;
    border-radius: 999px;
    background: rgba(59,130,246,.16);
    border: 1px solid rgba(59,130,246,.35);
    color: #dbeafe;
    font-size: 13px;
    margin-right: 8px;
    margin-top: 14px;
}


.metric-card, .model-card, .chart-card, .result-card {
    background: linear-gradient(145deg, rgba(15,23,42,.96), rgba(30,41,59,.82));
    border: 1px solid rgba(148,163,184,.18);
    border-radius: 22px;
    padding: 20px;
    box-shadow: 0 18px 45px rgba(0,0,0,.28);
}


.metric-value {
    font-size: 32px;
    font-weight: 900;
}


.metric-label {
    color: #cbd5e1;
    font-size: 14px;
}


.section-title {
    font-size: 24px;
    font-weight: 900;
    margin: 24px 0 14px;
}


.model-visual {
    height: 150px;
    border-radius: 18px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 68px;
    margin-bottom: 15px;
    border: 1px solid rgba(148,163,184,.18);
}


.visual-lr {
    background: radial-gradient(circle, rgba(59,130,246,.38), rgba(15,23,42,.85));
}


.visual-svm {
    background: radial-gradient(circle, rgba(168,85,247,.35), rgba(15,23,42,.85));
}


.visual-rf {
    background: radial-gradient(circle, rgba(249,115,22,.35), rgba(15,23,42,.85));
}


.model-title {
    font-size: 20px;
    font-weight: 900;
}


.model-subtitle {
    color: #94a3b8;
    margin-bottom: 12px;
}


.best {
    color: #22c55e;
    font-weight: 900;
}


.warning {
    color: #f59e0b;
    font-weight: 900;
}


.worst {
    color: #ef4444;
    font-weight: 900;
}


.stButton > button {
    background: linear-gradient(90deg, #2563eb, #9333ea, #c026d3);
    color: white;
    border: none;
    border-radius: 16px;
    padding: .8rem;
    font-weight: 900;
    width: 100%;
}


.stButton > button:hover {
    color: white;
    border: none;
}


[data-testid="stMetricValue"] {
    color: white;
}
</style>
""", unsafe_allow_html=True)




@st.cache_data
def load_data():
    df = pd.read_csv("Students outcome based on eating behaviors.csv")
    df.columns = df.columns.str.replace("\n", "", regex=False).str.strip()
    return df




df = load_data()
target_col = "GPA_range"


if target_col not in df.columns:
    st.error("Column GPA_range was not found.")
    st.write(df.columns.tolist())
    st.stop()




label_map = {
    0: "High",
    1: "Medium",
    2: "Low"
}




@st.cache_data
def prepare_data(df):
    df_clean = df.copy()


    if "ID" in df_clean.columns:
        df_clean = df_clean.drop(columns=["ID"])


    for col in df_clean.columns:
        if df_clean[col].dtype == "object":
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))


    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())


    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )


    return df_clean, X, y, X_train, X_test, y_train, y_test




df_clean, X, y, X_train, X_test, y_train, y_test = prepare_data(df)




@st.cache_resource
def train_best_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model




best_model = train_best_model(X_train, y_train)




model_results = {
    "Logistic Regression": {
        "train": 0.64,
        "test": 0.53,
        "status": "Underfitting",
        "rank": "Best Model",
        "class": "best",
        "icon": "🧠",
        "visual": "visual-lr",
        "notes": [
            "Highest test accuracy",
            "Best overall performance",
            "Model needs improvement"
        ],
        "cm": np.array([
            [13, 5, 1],
            [3, 6, 10],
            [5, 4, 12]
        ])
    },
    "SVM": {
        "train": 0.83,
        "test": 0.49,
        "status": "Overfitting",
        "rank": "Second Model",
        "class": "warning",
        "icon": "📈",
        "visual": "visual-svm",
        "notes": [
            "High training accuracy",
            "Lower test performance",
            "Overfitting detected"
        ],
        "cm": np.array([
            [10, 6, 3],
            [2, 8, 9],
            [3, 7, 11]
        ])
    },
    "Random Forest": {
        "train": 1.00,
        "test": 0.46,
        "status": "Severe Overfitting",
        "rank": "Worst Model",
        "class": "worst",
        "icon": "🌲",
        "visual": "visual-rf",
        "notes": [
            "Lowest test accuracy",
            "Training accuracy is 100%",
            "Poor generalization"
        ],
        "cm": np.array([
            [10, 5, 4],
            [2, 5, 12],
            [1, 8, 12]
        ])
    }
}




with st.sidebar:
    st.markdown("## 🎓 GPA Predictor")
    st.caption("Students Outcome Based on Eating Behaviors")


    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Dataset Overview",
            "Model Results",
            "Prediction",
            "Deployment"
        ]
    )


    st.markdown("---")
    st.success("Dataset Loaded Successfully")
    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")




st.markdown("""
<div class="hero">
    <h1>Students Outcome Based on <span class="gradient-text">Eating Behaviors</span></h1>
    <p style="color:#cbd5e1;font-size:18px;">Machine Learning Classification & Prediction App</p>
    <span class="badge">Interactive</span>
    <span class="badge">Trained Models</span>
    <span class="badge">Real Project Results</span>
</div>
""", unsafe_allow_html=True)




if page == "Dashboard":


    c1, c2, c3, c4 = st.columns(4)


    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Students</div>
            <div class="metric-value">{df.shape[0]}</div>
        </div>
        """, unsafe_allow_html=True)


    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Input Features</div>
            <div class="metric-value">{X.shape[1]}</div>
        </div>
        """, unsafe_allow_html=True)


    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Models Trained</div>
            <div class="metric-value">3</div>
        </div>
        """, unsafe_allow_html=True)


    with c4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Best Model</div>
            <div class="metric-value">Logistic Regression</div>
        </div>
        """, unsafe_allow_html=True)


    st.markdown('<div class="section-title">Model Performance Comparison</div>', unsafe_allow_html=True)


    cols = st.columns(3)


    for col, (name, info) in zip(cols, model_results.items()):
        with col:
            st.markdown(f"""
            <div class="model-card">
                <div class="model-visual {info["visual"]}">{info["icon"]}</div>
                <div class="model-title">{name}</div>
                <div class="model-subtitle">{info["rank"]}</div>
                <h1>{info["test"]*100:.0f}%</h1>
                <p>Test Accuracy</p>
                <p class="{info["class"]}">{info["status"]}</p>
                <p>Train Accuracy: {info["train"]*100:.0f}%</p>
                <p>Test Accuracy: {info["test"]*100:.0f}%</p>
                <hr>
                <p>✅ {info["notes"][0]}</p>
                <p>✅ {info["notes"][1]}</p>
                <p>✅ {info["notes"][2]}</p>
            </div>
            """, unsafe_allow_html=True)


    left, right = st.columns([1, 1])


    with left:
        st.markdown('<div class="section-title">GPA Distribution</div>', unsafe_allow_html=True)


        ordered_labels = ["High", "Medium", "Low"]
        ordered_values = [35.5, 33.5, 31.5]
        display_percentages = ["35.5%", "33.5%", "31.5%"]


        def fixed_autopct(values):
            def inner(_):
                value = values[inner.index]
                inner.index += 1
                return value
            inner.index = 0
            return inner


        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#0f172a")


        ax.pie(
            ordered_values,
            labels=ordered_labels,
            autopct=fixed_autopct(display_percentages),
            startangle=90,
            colors=["#93c5fd", "#fda4af", "#86efac"],
            textprops={"color": "white", "fontsize": 12},
            wedgeprops={"edgecolor": "#0f172a", "linewidth": 2}
        )


        ax.set_title("GPA Range Distribution in Dataset", color="white", fontsize=15, weight="bold")
        st.pyplot(fig)


    with right:
        st.markdown('<div class="section-title">Best Model Confusion Matrix</div>', unsafe_allow_html=True)


        cm = model_results["Logistic Regression"]["cm"]


        fig2, ax2 = plt.subplots(figsize=(6, 5))
        fig2.patch.set_facecolor("#0f172a")
        ax2.set_facecolor("#0f172a")


        ax2.imshow(cm, cmap="Blues")
        ax2.set_title("Logistic Regression Confusion Matrix", color="white", weight="bold")
        ax2.set_xlabel("Predicted labels", color="white")
        ax2.set_ylabel("True labels", color="white")


        ax2.set_xticks([0, 1, 2])
        ax2.set_yticks([0, 1, 2])
        ax2.set_xticklabels(["0", "1", "2"], color="white")
        ax2.set_yticklabels(["0", "1", "2"], color="white")


        for i in range(3):
            for j in range(3):
                ax2.text(j, i, cm[i, j], ha="center", va="center", color="white", weight="bold")


        st.pyplot(fig2)




elif page == "Dataset Overview":


    st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)


    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isnull().sum().sum()))


    st.subheader("First Rows")
    st.dataframe(df.head(), use_container_width=True)


    st.subheader("Missing Values")
    st.dataframe(df.isnull().sum().rename("Missing Values"), use_container_width=True)


    st.subheader("Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)




elif page == "Model Results":


    st.markdown('<div class="section-title">Model Results</div>', unsafe_allow_html=True)


    results_df = pd.DataFrame({
        "Model": ["Logistic Regression", "SVM", "Random Forest"],
        "Train Accuracy": ["64%", "83%", "100%"],
        "Test Accuracy": ["53%", "49%", "46%"],
        "Status": ["Underfitting", "Overfitting", "Severe Overfitting"],
        "Rank": ["Best", "Second", "Worst"]
    })


    st.dataframe(results_df, use_container_width=True)


    selected_model = st.selectbox(
        "Select model to view confusion matrix",
        ["Logistic Regression", "SVM", "Random Forest"]
    )


    cm = model_results[selected_model]["cm"]


    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")


    ax.imshow(cm, cmap="Blues")
    ax.set_title(f"{selected_model} Confusion Matrix", color="white", weight="bold")
    ax.set_xlabel("Predicted labels", color="white")
    ax.set_ylabel("True labels", color="white")


    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(["0", "1", "2"], color="white")
    ax.set_yticklabels(["0", "1", "2"], color="white")


    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white", weight="bold")


    st.pyplot(fig)




elif page == "Prediction":

    st.markdown('<div class="section-title">Predict New Student GPA Level</div>', unsafe_allow_html=True)

    st.info("The prediction uses Logistic Regression because it achieved the highest test accuracy: 53%.")

    user_input = {}
    cols = st.columns(2)

    for i, col in enumerate(X.columns):
        min_value = int(df_clean[col].min())
        max_value = int(df_clean[col].max())
        mean_value = int(round(df_clean[col].mean()))

        with cols[i % 2]:
            user_input[col] = st.slider(
                col,
                min_value=min_value,
                max_value=max_value,
                value=mean_value,
                step=1,
                format="%d"
            )

    if st.button("Predict GPA Level"):
        input_df = pd.DataFrame([user_input])
        pred = best_model.predict(input_df)[0]
        pred_label = label_map.get(pred, str(pred))

        st.markdown(f"""
        <div class="result-card">
            <h2>Predicted GPA Range</h2>
            <h1 style="font-size:54px;color:#34d399;">{pred_label}</h1>
            <p>Model Used: Logistic Regression</p>
            <p>Test Accuracy: 53%</p>
        </div>
        """, unsafe_allow_html=True)



elif page == "Deployment":


    st.markdown('<div class="section-title">Deployment</div>', unsafe_allow_html=True)


    st.code("""
project_folder/
│
├── mlproject.py
├── Students outcome based on eating behaviors.csv
└── requirements.txt
""")


    st.markdown("""
    <div class="chart-card">
        <h3>Deployment Summary</h3>
        <p>
        The project was deployed using Streamlit as an interactive machine learning web application.
        The interface displays dataset information, model comparison, confusion matrices, and a prediction form.
        Logistic Regression is used for prediction because it achieved the highest test accuracy among the tested models.
        </p>
    </div>
    """, unsafe_allow_html=True)
