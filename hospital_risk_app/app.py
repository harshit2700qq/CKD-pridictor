import streamlit as st
import numpy as np
import joblib


model = joblib.load("models/model.pkl")
imputer = joblib.load("models/imputer.pkl")
encoders = joblib.load("models/encoders.pkl")
features = joblib.load("models/features.pkl")

st.set_page_config(page_title="CKD Health Check", layout="centered")


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.title(" Login")

    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user == "admin" and pwd == "1234":
            st.session_state.logged_in = True
        else:
            st.error("Wrong username or password")

if not st.session_state.logged_in:
    login()
    st.stop()


st.title(" Kidney Health Check")

st.markdown("### Please enter your details")

age = st.slider("Age", 1, 100, 40)
bp = st.slider("Blood Pressure", 60, 180, 80)
bgr = st.slider("Blood Sugar", 70, 400, 120)
bu = st.slider("Urea Level", 1, 150, 40)
sc = st.slider("Creatinine Level", 0.1, 15.0, 1.2)
hemo = st.slider("Hemoglobin", 5.0, 18.0, 14.0)


htn = st.radio("Do you have high blood pressure?", ["no", "yes"])
dm = st.radio("Do you have diabetes?", ["no", "yes"])
pe = st.radio("Swelling in legs?", ["no", "yes"])
ane = st.radio("Feeling weak or anemic?", ["no", "yes"])


htn = encoders["htn"].transform([htn])[0]
dm = encoders["dm"].transform([dm])[0]
pe = encoders["pe"].transform([pe])[0]
ane = encoders["ane"].transform([ane])[0]


input_data = {
    "age": age,
    "bp": bp,
    "bgr": bgr,
    "bu": bu,
    "sc": sc,
    "hemo": hemo,
    "htn": htn,
    "dm": dm,
    "pe": pe,
    "ane": ane
}


sample = []
for f in features:
    sample.append(input_data.get(f, 0))

if st.button("Check My Health"):

    sample = np.array([sample])
    sample = imputer.transform(sample)

    pred = model.predict(sample)[0]
    prob = model.predict_proba(sample)[0][0]

    st.markdown("## Result")

    
    if pred == 0:
        st.error(" Risk of Kidney Disease Detected")
    else:
        st.success(" Low Risk")

    
    st.write(f"### Risk Level: {round(prob*100,2)}%")

    
    if prob > 0.5:
        st.warning(" You should consult a doctor as soon as possible.")
        dialysis = "May require frequent dialysis (8–16/month)"
    else:
        st.info(" You are mostly safe, but regular checkups are recommended.")
        dialysis = "No dialysis needed currently"

    st.write(f"### Dialysis Suggestion: {dialysis}")

    
    if sc > 5 or bu > 100:
        st.error("python train_ Critical values detected! Visit a doctor immediately!")