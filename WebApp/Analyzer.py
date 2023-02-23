import sys
import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
import plotly.colors as pcolors
from models.classifiers import DummyClassifier
from models.classifiers.SimpleRusUkrWarRelevanceClassifier import SimpleRusUkrWarRelevanceClassifier
from models.classifiers.NarrativeRecognitionClassifier import NarrativeRecognitionClassifier
from models.classifiers.MisinformationClassifier import MisinformationClassifier
from models.classifiers.MisinformationSimCSEClassifier import MisinformationSimCSEClassifier
from models.BotDetector import BotDetector
from models.ClassifierSummarizer import ClassifierSummarizer
from apputils import set_color

st.set_page_config(page_title="Analyzer")

models_to_run = ["RusUkrWarRelevance","narrativeRecognition","misInformationRoberta","misInformationSimCSE","botDetector"]

# cache imports of text
@st.cache_data
def imports():
   from propaganda import base_disinfo
   return base_disinfo

base_disinfo =  imports()

# to run on localhost only: --server.address=127.0.0.1

primary = 'rgb(0,135,131)'
neg = 'rgb(239, 123, 0)'


# Set session variables
if 'fig' not in st.session_state:
    st.session_state['fig'] = None
if 'fig2' not in st.session_state:
   st.session_state['fig2'] = None
if "i" not in st.session_state:
   st.session_state["i"] = 0
if "disinfo" not in st.session_state:
   st.session_state["disinfo"] = base_disinfo
if "initialRun" not in st.session_state:
   st.session_state["initialRun"] = False
if "text" not in st.session_state:
   st.session_state["text"] = None
if "mode" not in st.session_state:
   st.session_state["mode"] = "single"
if "score" not in st.session_state:
   st.session_state["score"] = None

i = st.session_state.i
@st.cache_resource
def loadDummy():
   return DummyClassifier()

@st.cache_resource
def loadRusUkrWarRelevance():
   return SimpleRusUkrWarRelevanceClassifier()

@st.cache_resource
def loadNarrativeRecognition():
   return NarrativeRecognitionClassifier()

@st.cache_resource
def loadMisInformationRoberta():
   return MisinformationClassifier()

@st.cache_resource
def loadBotDetector():
   return BotDetector()

@st.cache_resource
def loadClassifierSummarizer():
   return ClassifierSummarizer()

@st.cache_resource
def loadMisInformationSimCSE():
   return MisinformationSimCSEClassifier()

# using loader functions to load models to cache - avoids having to reload models for each refresh
# declare model loaders and name of metrics
model_loaders = {"RusUkrWarRelevance":loadRusUkrWarRelevance,"narrativeRecognition":loadNarrativeRecognition,
                "misInformationRoberta":loadMisInformationRoberta, "botDetector":loadBotDetector,"misInformationSimCSE":loadMisInformationSimCSE}
model_metrics = {"RusUkrWarRelevance":"Relevant to <br> Rus/Ukr war",
             "narrativeRecognition":"Matching <br> known <br> propaganda <br> narrative", "misInformationRoberta":"Misinformation Classification <br> (RoBERTa)",
              "misInformationSimCSE":"Misinformation Classifier <br> (SimCSE)", "botDetector":"Potential Bot?"}

# function to be run when analyse button is pressed
def run(config:list[str]):
   total_score_dict = {}
   summarizer = loadClassifierSummarizer()
   text = st.session_state.text
   st.session_state.initialRun = True
   global fig
   global fig2
   fig = go.Figure()
   fig2 = go.Figure()
   count = 0

   narrative_analysis = None
   for model_name in config:
      # my_bar.progress(count,text=f"Running {model_name}")
      count += 25
      model = model_loaders[model_name]()
      if model_name == "narrativeRecognition":
         narratives = st.session_state.disinfo.split(";")
         out = model.predict([text],narratives)
         updatePlot(out["maxconf"][0],model_metrics[model_name])
         narrative_analysis = out["confs"][0].copy()
         total_score_dict[model_name] = out["maxconf"][0]
      elif model_name == "misInformationRoberta":
         out = model.predict([text])["confs"][0][1]
         updatePlot(out,model_metrics[model_name])
         total_score_dict[model_name] = out
      elif model_name == "botDetector":
         if st.session_state.mode == "single":
            #total_score_dict[model_name] = 0.5
            pass
         else:
            out = model.predict(file["date"].iloc[st.session_state.i],file["user"].iloc[st.session_state.i])[0]
            updatePlot(out,model_metrics[model_name])
            total_score_dict[model_name] = out
      elif model_name == "RusUkrWarRelevance":
         out = model.predict([text])["confs"][0][0]
         updatePlot(out,model_metrics[model_name])
         total_score_dict[model_name] = out
      elif model_name == "misInformationSimCSE":
         out = model.predict([text])["confs"][0][1]
         updatePlot(out,model_metrics[model_name])
         total_score_dict[model_name] = out

   st.session_state.fig = fig
   st.session_state.score = np.round(summarizer.predict(total_score_dict)*100,1)

   if narrative_analysis is not None:
      labels = []
      for nar in narratives:
         if len(nar)>73:
            labels.append(nar[0:70]+"...")
         else:
            labels.append(nar)

      updateNarrativePlot(narrative_analysis,labels)
      st.session_state.fig2 = fig2

# add analyses to plot
def updatePlot(value,label):
   global fig
   fig.add_trace(go.Bar(x=[label], y=[value],
                  base=[0],marker_color=set_color(value)))

   fig.update_yaxes(range=[0, 1])
   fig.layout.update(showlegend=False)
   fig.update_coloraxes()

def updateNarrativePlot(narrative_analysis,labels):
   global fig2
   fig2 = go.Figure()
   fig2.add_trace(go.Bar(x=narrative_analysis, y=labels,orientation='h',marker_color=primary))
   fig2.update_xaxes(range=[0,1])


def restMultiple():
   st.session_state.i = 0
   st.session_state.fig = None
   st.session_state.fig2 = None
   st.session_state.initialRun = False
   st.session_state.text = None

def clickNext():
   st.session_state.i +=1
   if st.session_state.i >= len(file.index):
      st.session_state.i = len(file.index) - 1
   st.session_state.text = file["content"].iloc[st.session_state.i]
   run(models_to_run)


def clickPrev():
   st.session_state.i -=1
   if st.session_state.i < 0:
      st.session_state = 0
   st.session_state.text = file["content"].iloc[st.session_state.i]
   run(models_to_run)


st.sidebar.title("STADFEST")

# Declare tabs
tab_analyze, tab_settings= st.tabs(["Analyzer", "Settings"])

# Settings for the analyzer
with tab_settings:

   # st.write("**Choose analyses to run:**")

   # col1, col2 = st.columns(2)

   # with col1:
   #    op1_chechbox = st.checkbox("Hashtag Analysis", value=True)
   #    op2_checkbox = st.checkbox("Sentiment Analysis", value=True)

   # with col2:
   #    op3_checkbox = st.checkbox("Propaganda recognizer", value=True)

   st.write("**Propaganda recognizer semantic input**")

   st.session_state.disinfo = st.text_area("Known disinformation",
   value=st.session_state.disinfo,height=200,help="Enter information to be run against inputs with semantic AI analysis. Separate each input with a semi colon (;)")


   col1,col2,col3 = st.columns(3)

   with col2:
      st.button("Update",use_container_width=True)

# Main page for the analyzer
with tab_analyze:
   st.session_state.text = st.text_area("Enter you text here")
   st.write("*or*")
   file = st.file_uploader("Upload a csv file",type={"csv","txt"},on_change=restMultiple)
   st.session_state.mode = "single"
   if file is not None:
      file = pd.read_csv(file)
      st.session_state.mode = "multiple"
      st.session_state.text = file["content"].iloc[st.session_state.i]

   col1,col2,col3 = st.columns(3)

   with col1:
      pass

   with col2:
      st.button("Run analysis", on_click=run,args=([models_to_run]),use_container_width=True)

   with col3:
      pass

   # my_bar = st.progress(0)
   st.subheader("Disinformation Metrics")

   if st.session_state.mode == "multiple":
      total = len(file.index)
      if st.session_state.initialRun:
         col1, col2, col3, col4, col5 = st.columns(5)

         with col5:
            st.button("Next",on_click=clickNext,use_container_width=True)

         with col2:
            st.write(f"({st.session_state.i+1}/{total})")

         with col1:
            st.button("Previous",on_click=clickPrev,use_container_width=True)

         cont = st.container()
         with cont:
            st.write(st.session_state.text)

            col1,col2 = st.columns([4,1])
            with col1:
               if st.session_state.fig is not None:
                  plot = st.plotly_chart(st.session_state.fig, use_container_width=True)
            with col2:
               st.metric("Total Score",value=st.session_state.score)

            if st.session_state.fig2 is not None:
               plot2 = st.plotly_chart(st.session_state.fig2,use_container_width=True)

            else:
               plot = st.empty()
               plot2 = st.empty()



   else:
      st.write(st.session_state.text )
      col1,col2 = st.columns([4,1])
      with col1:
         if st.session_state.fig is not None:
            plot = st.plotly_chart(st.session_state.fig, use_container_width=True)
      with col2:
         st.metric("Total Disinformation Score",value=st.session_state.score)

      if st.session_state.fig2 is not None:
         plot2 = st.plotly_chart(st.session_state.fig2,use_container_width=True)

      else:
         plot = st.empty()
         plot2 = st.empty()









