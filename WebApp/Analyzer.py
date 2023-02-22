import sys
sys.path.append("/home/local/NTU/eijor/hackathon2023/disinformation-analyser")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pcolors
from models.classifiers import DummyClassifier
from models.classifiers.SimpleUkraineSupportClassifier import SimpleUkraineSupportClassifier
from Apputils import set_color

st.set_page_config(page_title="Analyzer")

# cache imports of text
@st.cache_data
def imports():
   from facts_and_propaganda import disinfo,factualinfo
   return disinfo,factualinfo

disinfo,factualinfo = imports()

# to run on localhost only: --server.address=127.0.0.1

primary = 'rgb(0,135,131)'
neg = 'rgb(239, 123, 0)'

# Set session variables
if 'fig' not in st.session_state:
    st.session_state['fig'] = None
if "i" not in st.session_state:
   st.session_state["i"] = 0

i = st.session_state.i
@st.cache_resource
def loadDummy():
   return DummyClassifier()

@st.cache_resource
def loadUkraineSupport():
   return SimpleUkraineSupportClassifier()

# declare model loaders and name of metrics
model_loaders = {"dummy":loadDummy, "UkraineSupport":loadUkraineSupport}
model_metrics = {"dummy": "Something", "UkraineSupport":"Ukraine <br> Support <br> Sentiment"}

# function to be run when analyse button is pressed
def run(config:list[str],text):
   global fig
   fig = go.Figure()
   count = 0

   for model_name in config:
      # my_bar.progress(count,text=f"Running {model_name}")
      count += 25
      model = model_loaders[model_name]()
      out = model.predict([text])["confs"][0][0]
      updatePlot(out,model_metrics[model_name])
   
   st.session_state.fig = fig

# add analyses to plot
def updatePlot(value,label):
   global fig
   fig.add_trace(go.Bar(x=[label], y=[value],
                  base=[0],marker_color=set_color(value)))
      
   fig.update_yaxes(range=[0, 1])
   fig.layout.update(showlegend=False)
   fig.update_coloraxes()

def restMultiple():
   st.session_state.i = 0


st.sidebar.title("STADFEST")

# Declare tabs
tab_analyze, tab_settings= st.tabs(["Analyzer", "Settings"])

# Settings for the analyzer
with tab_settings:
   
   st.write("**Choose analyses to run:**")

   col1, col2 = st.columns(2)

   with col1:
      op1_chechbox = st.checkbox("Hashtag Analysis", value=True)
      op2_checkbox = st.checkbox("Sentiment Analysis", value=True)

   with col2:
      op3_checkbox = st.checkbox("Propaganda recognizer", value=True)

   st.write("**Propaganda recognizer semantic input**")
   
   col1, col2 = st.columns(2)
   with col1:
      disinfo = st.text_area("Known disinformation", disabled=(not op3_checkbox),
      value=disinfo,height=200,help="Enter information to be run against inputs with semantic AI analysis. Separate each input with a semi colon (;)")

   with col2:
      facts = st.text_area("Known factual information",disabled=(not op3_checkbox),
      value=factualinfo,height=200)

   col1,col2,col3 = st.columns(3)

   with col2:
      st.button("Update",use_container_width=True)

# Main page for the analyzer
with tab_analyze:
   text = st.text_area("Enter you text here")
   st.write("*or*")
   file = st.file_uploader("Upload a csv file",type={"csv","txt"})
   mode = "single"
   if file is not None:
      file = pd.read_csv(file)
      mode = "multiple"
      text = file["content"].iloc[0]

   col1,col2,col3 = st.columns(3)

   with col1:
      pass

   with col2:
      st.button("Run analysis", on_click=run,args=(["dummy","UkraineSupport","dummy"],text),use_container_width=True)

   with col3:
      pass

   # my_bar = st.progress(0)
   st.subheader("Disinformation Metrics")

   if mode == "multiple":
      total = len(file.index)
      cont = st.container()
      with cont:
         if st.session_state.i < total-1:
            st.session_state.i += 1
            text = file["content"].iloc[st.session_state.i+1]
            st.button("Next",on_click=run,args=(["dummy","UkraineSupport","dummy"],text))
         else:
            st.write("Reached end of data")

         if st.session_state.fig is not None:
            plot = st.plotly_chart(st.session_state.fig, use_container_width=True)
            st.write(f"({st.session_state.i+1}/{total+1})")
            st.markdown("---")
            st.write(file["content"].iloc[st.session_state.i])
            st.markdown("---")
         else:
            plot = st.empty()




   else:
      if st.session_state.fig is not None:
         plot = st.plotly_chart(st.session_state.fig, use_container_width=True)
            
         st.markdown("---")
         st.write(text)
         st.markdown("---")
      else:
         plot = st.empty()





   group_labels = ['Hashtags', 'Pro Ukranian Sentiment', 'Disinformation Narrative Recognition','Likness to known Russian Propaganda','Model 5']




