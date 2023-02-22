import streamlit as st
import numpy as np

st.sidebar.title("STADFEST")

st.header("Live Monitoring")

tab_live, tab_control= st.tabs(["Live Feed", "Control Panel"],)


with tab_live:

   with st.container():
      st.write("This is inside the container")

      # You can call any Streamlit command, including custom components:
      st.bar_chart(np.random.randn(50, 3))

with tab_control:
   st.subheader("Choose feeds to monitor")
   col1, col2  = st.columns(2,gap="large")
   with col1:
      twitterRUS_checkbox = st.checkbox("Twitter: Russia", value=False)
      twitterDONBAS_checkbox = st.checkbox("Twitter: Donbas", value=False)
   with col2:
      cnn_checkbox = st.checkbox("CNN World News", value=False)
      RT_world_checkbox = st.checkbox("RT: World News", value=False)
      RT_russia_checkbox = st.checkbox("RT: Russia", value=False)

   confidence_threshold = st.slider("Confidence threshold", min_value=0.2, max_value=0.99, value=0.7)


