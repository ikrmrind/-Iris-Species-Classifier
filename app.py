import streamlit as st
import pandas as pd
import joblib

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Iris Flower Predictor",
    page_icon="🌸",
    layout="centered"  # Changed to centered to bring everything into the middle
)

# =========================
# LOAD MODEL
# =========================
try:
    model = joblib.load("Iris_DataSet.pkl")
    columns = joblib.load("Columns.pkl")
except Exception as e:
    st.error(f"Error loading model or columns: {e}")

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
.main {
    background-color: #f8fafc;
}

.title {
    font-size: 38px;
    font-weight: 700;
    text-align: center;
    color: #1e293b;
    margin-bottom: 0px;
}

.subtitle {
    text-align: center;
    color: #64748b;
    font-size: 16px;
    margin-bottom: 30px;
}

.pred-box {
    background: #ffffff;
    padding: 15px;
    border-radius: 12px;
    border-left: 5px solid #16a34a;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    text-align: center;
    margin-top: 25px;
}

.pred-box h3 {
    font-size: 16px;
    color: #64748b;
    margin-bottom: 5px;
}

.pred-box h2 {
    font-size: 24px;
    color: #16a34a;
    margin-top: 0px;
}

div[data-testid="stCaptionContainer"] {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown(
    '<div class="title">🌸 Iris Flower Species Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning Powered Flower Classification System</div>',
    unsafe_allow_html=True
)

# =========================
# LAYOUT (Inputs Left, Small Box Right)
# =========================
# Using a 2:1 ratio keeps inputs centered and prominent while putting the result in a smaller right box
left, right = st.columns([2, 1], gap="large")

with left:
    st.subheader("📥 Enter Measurements")

    sepal_length = st.number_input(
        "Sepal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=5.1
    )

    sepal_width = st.number_input(
        "Sepal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=3.5
    )

    petal_length = st.number_input(
        "Petal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=1.4
    )

    petal_width = st.number_input(
        "Petal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=0.2
    )

    predict_btn = st.button(
        "🔍 Predict Species",
        use_container_width=True
    )

# =========================
# PREDICTION (Displays on Upper Right)
# =========================
with right:
    if predict_btn:
        input_data = pd.DataFrame(
            [[
                sepal_length,
                sepal_width,
                petal_length,
                petal_width
            ]],
            columns=columns
        )

        prediction = model.predict(input_data)[0]

        # Renders directly in the top-right column without scrolling
        st.markdown(
            f"""
            <div class="pred-box">
                <h3>✅ Result</h3>
                <h2>{prediction}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        # Placeholder styling before clicking predict
        st.markdown(
            """
            <div style="background: #f1f5f9; padding: 15px; border-radius: 12px; text-align: center; margin-top: 50px; color: #94a3b8; font-size: 14px;">
                👈 Enter values and click <b>Predict Species</b> to see results here.
            </div>
            """, 
            unsafe_allow_html=True
        )

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption(
    "Built with Streamlit | Machine Learning Project by Ikram"
)