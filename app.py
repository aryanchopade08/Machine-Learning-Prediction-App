import joblib
import streamlit as st

classification_model = joblib.load("best_classification_model.pkl")
regression_model = joblib.load("best_regression_model.pkl")

st.set_page_config(page_title="ML Prediction App", layout="wide")

st.title("Machine Learning Prediction App")

problem = st.selectbox(
    "Select Problem Type",
    ["Classification", "Regression"]
)

st.write("Selected Problem:", problem)

if problem == "Classification":
    algorithm = st.selectbox(
        "Select Classification Algorithm",
        ["KNN", "Naive Bayes"]
    )

else:

    year = st.number_input("Year", 2000, 2035, 2018)
    mileage = st.number_input("Mileage", 0)
    tax = st.number_input("Tax", 0)
    mpg = st.number_input("MPG", 0.0)
    engine = st.number_input("Engine Size", 0.0)

    if st.button("Predict"):

        # Total features = 32
        data = [0] * 33

        data[0] = year
        data[1] = mileage
        data[2] = tax
        data[3] = mpg
        data[4] = engine

        prediction = regression_model.predict([data])

        st.success(f"Predicted Price: {prediction[0]:.2f}")


st.subheader("Enter Input Values")

if problem == "Classification":

    age = st.number_input("Age", 0)
    restingbp = st.number_input("Resting BP", 0)
    cholesterol = st.number_input("Cholesterol", 0)
    fastingbs = st.number_input("Fasting BS (0/1)", 0, 1)
    maxhr = st.number_input("Max Heart Rate", 0)
    oldpeak = st.number_input("Oldpeak", 0.0)

    sex_m = st.selectbox("Sex", [0,1])

    cp_ata = st.selectbox("ChestPainType_ATA", [0,1])
    cp_nap = st.selectbox("ChestPainType_NAP", [0,1])
    cp_ta = st.selectbox("ChestPainType_TA", [0,1])

    ecg_normal = st.selectbox("RestingECG_Normal", [0,1])
    ecg_st = st.selectbox("RestingECG_ST", [0,1])

    angina = st.selectbox("ExerciseAngina_Y", [0,1])

    slope_flat = st.selectbox("ST_Slope_Flat", [0,1])
    slope_up = st.selectbox("ST_Slope_Up", [0,1])

    if st.button("Predict"):

        prediction = classification_model.predict([[
            age,
            restingbp,
            cholesterol,
            fastingbs,
            maxhr,
            oldpeak,
            sex_m,
            cp_ata,
            cp_nap,
            cp_ta,
            ecg_normal,
            ecg_st,
            angina,
            slope_flat,
            slope_up
        ]])

        st.success(f"Prediction: {prediction[0]}")