import streamlit as st

page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
background-image: url("https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExNHc0dThjc3lvNDJudXpyNWk5ZnRqYmR0bTZva2ZwZm1waDgyazFzbiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3ov9k1173PdfJWRsoE/giphy.gif");
background-size: cover;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Title
st.title("Welcome to our Major Project!")

# Steps for the user
st.header("Steps to Train Your Models")
steps = [
    "1. Upload your dataset (CSV file).",
    "2. Select the models you want to train.",
    "3. Choose preprocessing options (e.g., scaling, handling missing values).",
    "4. Train the selected models.",
    "5. Analyze the results and choose the best model."
]
for step in steps:
    st.write(step)

# Get Started button
if st.button("Get Started"):
    st.switch_page("pages/app.py")
