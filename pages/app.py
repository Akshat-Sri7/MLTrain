import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time

gif_path = "ML_back.gif"  # Replace with your GIF file name or path

page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
background-image: url("https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExNHc0dThjc3lvNDJudXpyNWk5ZnRqYmR0bTZva2ZwZm1waDgyazFzbiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3ov9k1173PdfJWRsoE/giphy.gif");
background-size: cover;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)


# Custom CSS for transparent sidebar
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.08);  /* Adjust the last value for transparency */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize session state variables if not already present
if "results" not in st.session_state:
    st.session_state.results = []
if "best_model" not in st.session_state:
    st.session_state.best_model = None

# Sidebar for dataset upload, model selection, and preprocessing
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

st.sidebar.header("Model Selection")
selected_models = st.sidebar.multiselect("Select models to train",
                                         ["Logistic Regression", "Decision Tree", "Random Forest", "SVM",
                                          "K-Nearest Neighbors", "Neural Network"])

st.sidebar.header("Preprocessing Options")
scaling = st.sidebar.checkbox("Scale Features")
handle_missing = st.sidebar.checkbox("Handle Missing Values")

# Load dataset and apply basic preprocessing options
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview:")
    st.write(data.head())
    st.write("### Dataset Summary:")
    st.write(data.describe())
    st.divider()

    if handle_missing:
        data = data.dropna()

    if scaling:
        scaler = StandardScaler()
        data[data.columns] = scaler.fit_transform(data[data.columns])


# Define the neural network model
def create_neural_network(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Model training function
def train_model(model_name, X_train, X_test, y_train, y_test):
    start_time = time.time()
    if model_name == "Logistic Regression":
        model = LogisticRegression()
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_name == "Random Forest":
        model = RandomForestClassifier()
    elif model_name == "SVM":
        model = SVC()
    elif model_name == "K-Nearest Neighbors":
        model = KNeighborsClassifier()
    elif model_name == "Neural Network":
        model = create_neural_network(X_train.shape[1])
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        y_pred = (model.predict(X_test) > 0.5).astype("int32")
        training_time = time.time() - start_time
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")
        return {
            "model": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "time": training_time
        }

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    training_time = time.time() - start_time
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "time": training_time
    }


# Prepare dataset for training and store results in session state
if uploaded_file:
    X = data.drop(columns=["Class"])  # Replace "fraud" with your actual target column name
    y = data["Class"]
    if y.dtype != 'int' and y.nunique() < 20:
        y = y.astype('category').cat.codes

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    if st.sidebar.button("Train Selected Models"):
        if selected_models:
            st.session_state.results = []  # Reset results on new training
            for model_name in selected_models:
                result = train_model(model_name, X_train, X_test, y_train, y_test)
                st.session_state.results.append(result)
            st.session_state.best_model = max(st.session_state.results, key=lambda x: x['accuracy'])


# Helper functions for visualizations
def plot_comparison_chart(results_df):
    # Set the style
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size as needed

    # Plot a bar chart for accuracy only
    sns.barplot(data=results_df, x="model", y="accuracy", ax=ax, palette="viridis")

    # Customize the plot
    ax.set_title("Model Comparison by Accuracy", pad=20)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Model")
    ax.set_ylim(0, 1)  # Scale the y-axis to make differences more visible
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels

    # Add accuracy values on top of the bars as percentages rounded to two decimal places
    for p in ax.patches:
        accuracy_percentage = f'{p.get_height() * 100:.2f}%'
        ax.annotate(accuracy_percentage, (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points')

    # Remove the legend
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    # Show plot in Streamlit
    st.pyplot(fig)


def plot_training_time(results_df):
    fig, ax = plt.subplots()
    sns.lineplot(data=results_df, x="model", y="time", marker='o', ax=ax)

    ax.set_title("Training Time per Model")
    ax.set_ylabel("Time (seconds)")
    ax.set_xlabel("Model")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)


def plot_interactive_bar(results_df):
    # Interactive bar chart for metrics
    fig = px.bar(results_df, x="model", y=["accuracy", "precision", "recall", "f1"],
                 title="Interactive Model Comparison by Metrics",
                 labels={"value": "Metric Score", "variable": "Metrics"},
                 barmode='group')
    fig.update_layout(legend_title_text='Metric')
    st.plotly_chart(fig)


def plot_interactive_line(results_df):
    # Interactive line plot for training time
    fig = px.line(results_df, x="model", y="time", markers=True,
                  title="Training Time per Model",
                  labels={"time": "Training Time (seconds)", "model": "Model"})
    st.plotly_chart(fig)


def display_summary_table(results_df):
    results_df = results_df.style.background_gradient(cmap="viridis", subset=["accuracy", "precision", "recall", "f1"])
    st.write("### Summary Table:")
    st.dataframe(results_df)


# Display best model analysis only after "Show Best Model Analysis" button is clicked
if st.session_state.results:
    results_df = pd.DataFrame(st.session_state.results)

    st.write(f"## Best Model: {st.session_state.best_model['model']}")
    st.write(f"### Accuracy: {st.session_state.best_model['accuracy']*100:.4f} %")

    st.sidebar.write("***After Training Models click on the "
                     "Button below to get the Full analysis!***")

    if st.sidebar.button("Show Analysis"):
        st.divider()
        st.write("### Model Performance Comparison")
        plot_comparison_chart(results_df)
        plot_training_time(results_df)

        st.divider()
        st.write("### Interactive Metrics Comparison")
        plot_interactive_bar(results_df)
        plot_interactive_line(results_df)

        st.divider()
        display_summary_table(results_df)
