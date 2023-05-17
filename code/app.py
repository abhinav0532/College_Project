import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

st.title("Lead scoring dataset")

model = joblib.load("leads.C5")

# Total Time Spent on Website
total_time_spent = st.number_input("Total Time Spent on Website", value=0)

# Lead Origin_lead add form
lead_origin = st.number_input("Lead Origin_lead add form", value=0)

# Lead Source_direct traffic
lead_source_direct = st.number_input("Lead Source_direct traffic", value=0)

# Lead Source_welingak website
lead_source_welingak = st.number_input("Lead Source_welingak website", value=0)

# Do Not Email_yes
do_not_email = st.number_input("Do Not Email_yes", value=0)

# Last Activity_had a phone conversation
last_activity_phone = st.number_input("Last Activity_had a phone conversation", value=0)

# Last Activity_olark chat conversation
last_activity_chat = st.number_input("Last Activity_olark chat conversation", value=0)

# Last Activity_sms sent
last_activity_sms = st.number_input("Last Activity_sms sent", value=0)

# What is your current occupation_working professional
occupation_working = st.number_input("What is your current occupation_working professional", value=0)

# Last Notable Activity_unreachable
last_notable_activity = st.number_input("Last Notable Activity_unreachable", value=0)

features = np.array([total_time_spent,lead_origin,lead_source_direct,lead_source_welingak,do_not_email,last_activity_phone,last_activity_chat,last_activity_sms,occupation_working,last_notable_activity])

features = features.reshape(1,-1)
scale = StandardScaler()
scale.fit(features)


pred = model.predict(features)

if st.button("Predict"):
    if pred[0]==1:
        st.write("Lead found")
    else:
        st.write("lead not found")