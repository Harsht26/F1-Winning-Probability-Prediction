import streamlit as st
import pickle

# 1. Page Configuration
st.set_page_config(page_title="F1 Predictor", page_icon="🏎️")

# 2. Load the Model Data
@st.cache_resource
def load_data():
    # Make sure 'f1_model.pkl' is in the same folder
    with open("f1_model.pkl", "rb") as f:
        return pickle.load(f)

try:
    data_pack = load_data()
    model = data_pack["model"]
    ohe = data_pack["encoder"]
    all_features = data_pack["features"]
except FileNotFoundError:
    st.error("Error: 'f1_model.pkl' file not found! Please keep it in the same folder.")
    st.stop()

# 3. User Interface
st.title("🏎️ F1 Winner Probability Predictor")
st.markdown("Enter the race details to check winning chances.")

# Dynamic lists from your model features (to avoid manual typing errors)
# It extracts names like 'Max Verstappen' from 'Driver_Max Verstappen'
driver_list = sorted([col.replace('Driver_', '') for col in all_features if col.startswith('Driver_')])
team_list = sorted([col.replace('Team_', '') for col in all_features if col.startswith('Team_')])
race_list = sorted([col.replace('Race_', '') for col in all_features if col.startswith('Race_')])

# If lists are empty, use fallback defaults
if not driver_list: driver_list = ["Max Verstappen", "Lando Norris", "Lewis Hamilton"]
if not team_list: team_list = ["Red Bull", "McLaren", "Ferrari", "Mercedes"]
if not race_list: race_list = ["Bahrain Grand Prix", "Saudi Arabian Grand Prix"]

# Sidebar Inputs (all controls moved to left sidebar)
st.sidebar.header("Race Parameters")
grid = st.sidebar.slider("Starting Grid Position", 1, 20, 1)
rd = st.sidebar.number_input("Round Number", 1, 24, 1)
dr = st.sidebar.selectbox("Select Driver", driver_list)
gp = st.sidebar.selectbox("Select Grand Prix", race_list)
tm = st.sidebar.selectbox("Select Team", team_list)

# 4. Prediction Logic
if st.button("Predict Probability"):
    try:
        # Create input data with all features initialized to 0
        input_data = {col: [0] for col in all_features}

        # Set numeric features if present
        if 'Grid' in input_data:
            input_data['Grid'] = [grid]
        if 'Round' in input_data:
            input_data['Round'] = [rd]

        # Helper to set one-hot encoded feature robustly (handles mismatches/spaces)
        def set_one_hot(prefix, value):
            # try exact match first
            key = prefix + value
            if key in input_data:
                input_data[key] = [1]
                return True
            # try to find a column that starts with the prefix and contains the value
            for c in input_data:
                if c.startswith(prefix) and value.replace(' ', '') in c.replace(' ', ''):
                    input_data[c] = [1]
                    return True
            return False

        set_one_hot('Driver_', dr)
        set_one_hot('Team_', tm)
        set_one_hot('Race_', gp)

        # Convert to DataFrame for proper feature alignment
        import pandas as pd
        input_df = pd.DataFrame(input_data)

        # Make prediction
        prediction = model.predict_proba(input_df)[0][1]
        st.success(f"🏁 Winning Probability: {prediction*100:.2f}%")
    except KeyError as ke:
        st.error(f"Missing feature column: {ke}. Please verify model features.")
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

        