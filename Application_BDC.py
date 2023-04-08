#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 15:33:55 2023

@author: nicolasjulien
"""

import time  # to simulate a real time data, time loop

import requests
from bs4 import BeautifulSoup
from PIL import Image
import re  
import json as json 
from requests_html import HTMLSession 
from stqdm import stqdm
from datetime import datetime
from io import BytesIO
import urllib.request
from urllib.request import Request, urlopen
from io import StringIO
import pandas as pd
import requests

import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts

import sklearn
import joblib
import xgboost
import streamlit as st  # 🎈 data web app development

#------------------------------PRESENTATION DE LA PAGE----------------------------------------------

st.set_page_config(
    page_title="Challenge BDC ENSAE x MeilleurTaux",
    page_icon="https://drive.google.com/file/d/1rsobE8pEosOFjGyihHg6tN1oiqZQmwUV/view?usp=sharing",
    layout="wide",
)

st.title("Challenge BDC ENSAE x MeilleurTaux")

#------------------------------IMPORTATION DE LA BASE DE DONNEES TEST----------------------------------------------

#url = "https://drive.google.com/file/d/1PIdlpGqh8UoFYOUZCuE9kZ2ShEQTg3q1/view?usp=sharing"
#path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
#storage_options = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36'}
#data = pd.read_csv(path, storage_options=storage_options)
data = pd.read_csv('data_test.csv')

#------------------------------DEMANDE DE L'ADRESSE----------------------------------------------

adresse = st.text_input("Veuillez entrer l'adresse:")

if adresse != None:
    GEOCODE_URL = 'https://maps.googleapis.com/maps/api/geocode/json?address='+adresse+'&key='+st.secrets['gmaps_key']
    geo_response = requests.request("GET", GEOCODE_URL)
    geodata = json.loads(geo_response.text)
    try:
     lat_lon = pd.DataFrame([0, geodata['results'][0]['geometry']['location']['lat'], geodata['results'][0]['geometry']['location']['lng']],
                            columns=['lat', 'lon'])
     ville = geodata['results'][0]['address_components'][2]["long_name"]
    except IndexError:
     lat_lon = None
     ville = None
     st.write('Adresse non trouvée')

if lat_lon != None:
    st.write(lat_lon, ville)
    st.map(data=lat_lon)

# Input adresse
# API Google Maps
# Afficher la métropole/ type de bien
# charger le modele accordingly

#------------------------------IMPORTATION DU MODELE----------------------------------------------

pipe = joblib.load('Bordeaux-Metropole-Appartement-xgboost.joblib')
preprocessor = pipe[:-1]
st.write(preprocessor.named_steps, preprocessor.feature_names_in_)
xgb_model = pipe[-1]

#------------------------------INPUT DES CARACTERISTIQUES DU BIEN----------------------------------------------

echantillon = data.sample(1)
st.write(echantillon)

#------------------------------CREATION DE LA DONNEE ENTRANTE COMPLETE----------------------------------------------

data_echantillon = pd.DataFrame({'adresse_nom_voie':echantillon['adresse_nom_voie'],
                    'nom_commune':echantillon['nom_commune'],
                    'code_departement':echantillon['code_departement'],
                    'nombre_lots':echantillon['nombre_lots'],
                    'surface_reelle_bati':echantillon['surface_reelle_bati'],
                    'nombre_pieces_principales':echantillon['nombre_pieces_principales'],
                    'latitude':200,
                    'trimestre_vente':echantillon['trimestre_vente'],
                    'prix_m2_zone':echantillon['prix_m2_zone'],
                    'moyenne':echantillon['moyenne'],
                    'moyenne_brevet':echantillon['moyenne_brevet'],
                    'Banques':echantillon['Banques'],
                    'Bureaux_de_Poste':echantillon['Bureaux_de_Poste'],
                    'Commerces':echantillon['Commerces'],
                    'Ecoles':echantillon['Ecoles'],
                    'Collèges_Lycées':echantillon['Collèges_Lycées'],
                    'Medecins':echantillon['Medecins'],
                    'Gares':echantillon['Gares'],
                    'Cinema':echantillon['Cinema'],
                    'Bibliotheques':echantillon['Bibliotheques'],
                    'Espaces_remarquables_et_patrimoine':echantillon['Espaces_remarquables_et_patrimoine'],
                    'Taux_pauvreté_seuil_60':echantillon['Taux_pauvreté_seuil_60'],
                    'Q1':echantillon['Q1'],
                    'Mediane':echantillon['moyenne'],#attention il manque la mediane jsp pk
                    'Ecart_inter_Q_rapporte_a_la_mediane':echantillon['Ecart_inter_Q_rapporte_a_la_mediane'],
                    'D1':1000, #manque valeur
                    'D9':echantillon['D9'],
                    'Rapport_interdécile_D9/D1':float(echantillon['D9'])/1000,
                    'Gini':echantillon['Gini'],
                    'Part_revenus_activite':30,
                    'Part_salaire':echantillon['Part_salaire'],
                    'Part_revenus_chomage':echantillon['Part_revenus_chomage'],
                    'Part_revenus_non_salariées':echantillon['Part_revenus_non_salariées'],
                    'Part_retraites':echantillon['Part_retraites'],
                    'Part_revenus_patrimoine':echantillon['Part_revenus_patrimoine'],
                    'Part_prestations_sociales':20,
                    'Part_impôts':echantillon['Part_impôts']                 
    })

#------------------------------PREDICTION----------------------------------------------

prediction = pipe.predict(data_echantillon)
st.write(float(prediction))

# import matplotlib.pyplot as plt
# from xgboost import plot_tree
# plot_tree(xgb_model)
# st.pyplot(plot_tree(xgb_model,num_trees=0, rankdir='LR').figure)

#------------------------------BONUS---------------------------------------------

#Afficher l'adresse sur une Map
#Ajouter Feature Importance
#Ajouter Intervalle Confiance
