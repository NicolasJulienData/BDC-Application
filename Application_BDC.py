#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 15:33:55 2023

@author: nicolasjulien
"""

#------------------------------IMPORTS----------------------------------------------

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

#------------------------------DEMANDE DE L'ADRESSE----------------------------------------------

with st.sidebar:
    adresse = st.text_input("Veuillez entrer l'adresse:")

    if adresse != None:
        
        GEOCODE_URL = 'https://maps.googleapis.com/maps/api/geocode/json?address='+adresse+'&key='+st.secrets['gmaps_key']
        geo_response = requests.request("GET", GEOCODE_URL)
        geodata = json.loads(geo_response.text)
        try:
         lat_lon = pd.DataFrame({'lat':[geodata['results'][0]['geometry']['location']['lat']], 'lon':[geodata['results'][0]['geometry']['location']['lng']]})
         ville = geodata['results'][0]['address_components'][2]["long_name"]
         adresse_nom_voie = geodata['results'][0]['address_components'][1]["short_name"]
         code_departement = geodata['results'][0]['address_components'][6]["short_name"]
        except IndexError:
         lat_lon = None
         ville = None
         st.write('Adresse non trouvée')
        
        type_bien = st.selectbox("Sélectionner le type de bien",("Appartement", "Maison"))
        
        
#------------------------------IMPORTATION DE LA BASE DE DONNEES TEST----------------------------------------------

#url = "https://drive.google.com/file/d/1PIdlpGqh8UoFYOUZCuE9kZ2ShEQTg3q1/view?usp=sharing"
#path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
#storage_options = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36'}
#data = pd.read_csv(path, storage_options=storage_options)
data = pd.read_csv('data_test.csv')
data = data[(data['nom_commune']==ville)&(data['type_local']==type_bien)]

#------------------------------INPUT DES CARACTERISTIQUES DU BIEN----------------------------------------------

if (len(lat_lon)!=0) & (len(data)!=0):
    with st.sidebar:

        if (type_bien != None) & (ville != None):

            nombre_pieces_principales = st.slider('Nombre de pièces principales', min_value = int(min(data['nombre_pieces_principales'])),
                                                 max_value = int(max(data['nombre_pieces_principales'])), value = int(np.mean(data['nombre_pieces_principales'])), 
                                                 step = 1)
            surface_reelle_bati = st.slider('Surface réelle du batiment (en mètres carrés)', min_value = float(min(data['surface_reelle_bati'])),
                                                 max_value = float(max(data['surface_reelle_bati'])), value = float(np.mean(data['surface_reelle_bati'])))
            if type_bien == "Maison":
                surface_terrain = st.slider('Surface du terrain de la maison (en mètres carrés)', min_value = 0,
                                                 max_value = 10000, value=500)       

    st.map(data=lat_lon)

#------------------------------IMPORTATION DU MODELE----------------------------------------------
col1, col2 = st.columns(2)

if np.isin(ville,['Paris','Marseille','Lyon','Lille','Bordeaux','Toulouse','Nice','Nantes','Montpellier','Rennes']):

    pipe = joblib.load('{}-{}.joblib'.format(ville,type_bien))
    with col1:
    st.write('Modele Chargé pour la ville de {} pour un bien de type {}'.format(ville,type_bien))
    preprocessor = pipe[:-1]
#    st.write(preprocessor.named_steps, preprocessor.feature_names_in_)
    xgb_model = pipe[-1]
else:
    st.write("Ville non couverte par notre modèle. Veuillez réessayer dans l'une des métropoles suivantes : Paris, Marseille, Lyon, Lille, Bordeaux, Toulouse, Nice, Nantes, Montpellier, Rennes")

#------------------------------CREATION DE LA DONNEE ENTRANTE COMPLETE----------------------------------------------

echantillon = data.sample(1)
st.write(echantillon)

with st.sidebar:
    nombre_lots = st.slider('Nombre de pièces principales', min_value = int(min(data['nombre_lots'])),
                                                 max_value = int(max(data['nombre_lots'])), value = int(np.mean(data['nombre_lots'])), 
                                                 step = 1)
    trimestre_vente = st.selectbox(data['trimestre_vente'].unique)

if type_bien == 'Appartement':
    data_echantillon = pd.DataFrame({'adresse_nom_voie':adresse_nom_voie,
                        'nom_commune':ville,
                        'code_departement':code_departement,
                        'nombre_lots':nombre_lots,
                        'surface_reelle_bati':surface_reelle_bati,
                        'nombre_pieces_principales':nombre_pieces_principales,
                        'latitude':float(lat_lon['lat']),
                        'trimestre_vente':trimestre_vente,
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
                        'S80/S20':2,
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
    
elif type_bien =='Maison':
        data_echantillon = pd.DataFrame({'adresse_nom_voie':adresse_nom_voie,
                        'nom_commune':ville,
                        'code_departement':code_departement,
                        'nombre_lots':nombre_lots,
                        'surface_reelle_bati':surface_reelle_bati,
                        'nombre_pieces_principales':nombre_pieces_principales,
                        'surface_terrain':surface_terrain,
                        'latitude':float(lat_lon['lat']),
                        'trimestre_vente':trimestre_vente,
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
                        'Ecart_inter_Q_rapporte_a_la_mediane':echantillon['Ecart_inter_Q_rapporte_a_la_mediane'],
                        'D9':echantillon['D9'],
                        'Rapport_interdécile_D9/D1':float(echantillon['D9'])/1000,
                        'S80/S20':2,
                        'Gini':echantillon['Gini'],                      
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

with col2:
    st.write('Estimation du prix au mètre carré de votre bien immobilier : ', round(float(prediction),2), '€/mètre carré.')
    st.write('Estimation de la valeur de votre bien immobilier : ', int(float(prediction)*surface_reelle_bati), '€.')

# import matplotlib.pyplot as plt
# from xgboost import plot_tree
# plot_tree(xgb_model)
# st.pyplot(plot_tree(xgb_model,num_trees=0, rankdir='LR').figure)

#------------------------------BONUS---------------------------------------------

#Afficher l'adresse sur une Map
#Ajouter Feature Importance
#Ajouter Intervalle Confiance
