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

import warnings
warnings.filterwarnings('ignore')

import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import geopandas as gpd

import plotly.express as px  # interactive charts

import sklearn
from sklearn.neighbors import BallTree
from sklearn.linear_model import LinearRegression

import joblib
from joblib import Parallel, delayed

import xgboost
import streamlit as st  # 🎈 data web app development

#------------------------------PRESENTATION DE LA PAGE----------------------------------------------

st.set_page_config(
    page_title="Immobil.ia",
    page_icon="https://drive.google.com/file/d/1rsobE8pEosOFjGyihHg6tN1oiqZQmwUV/view?usp=sharing",
    layout="wide",
)

st.title("Immobil.ia - Business Data Challenge ENSAE x MeilleurTaux")

col_1, col_2 = st.columns(2)
with col_2:
    st.write('') #Saut de ligne
    st.write('') #Saut de ligne
    st.image("https://www.meilleurtaux.com/images_html/new-logo-mtx.svg", width = 270)
with col_1:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/LOGO-ENSAE.png/800px-LOGO-ENSAE.png", width = 180)
st.markdown("### Immobil.ia : l'application qui te permet d'estimer le prix de ton bien immobilier 🏠🏢") 

NoneType = type(None) # Création d'un NoneType pour les conditions

col1, col2 = st.columns(2) # Création d'autres colonnes par la suite

#------------------------------CHARGEMENT DES FONCTIONS DE PREPROCESSING----------------------------------------------
    
liste_equipements = [
                ['A203'],['A206'],
                ['B101','B102','B103','B201','B202','B203','B204','B205','B206'],
                ['C101','C102','C104','C105'],
                ['C201','C301','C302','C303','C304','C305'],['D201'],['E107','E108','E109'],['F303'],['F307'],['F313']
            ]
    
liste_var_garder=['id_mutation', 'date_mutation', 'numero_disposition', 'valeur_fonciere',
       'adresse_numero', 'adresse_nom_voie', 'adresse_code_voie',
       'code_commune', 'nom_commune', 'code_departement', 'LIBEPCI',
       'id_parcelle', 'nombre_lots', 'lot1_numero', 'lot1_surface_carrez',
       'lot2_numero', 'lot2_surface_carrez', 'lot3_numero',
       'lot3_surface_carrez', 'lot4_numero', 'lot4_surface_carrez',
       'lot5_numero', 'lot5_surface_carrez', 'type_local',
       'surface_reelle_bati', 'nombre_pieces_principales', 'surface_terrain',
       'longitude', 'latitude', 'geometry', 'quantile_prix', 'coeff_actu','prix_actualise','prix_m2_actualise','prix_m2','trimestre_vente','prix_m2_zone',
        'moyenne','moyenne_brevet','DCOMIRIS','indices', 'Banques', 'Bureaux_de_Poste', 'Commerces', 'Ecoles','Collèges_Lycées', 'Medecins',
       'Gares', 'Cinema', 'Bibliotheques', 'Espaces_remarquables_et_patrimoine', 'DCIRIS',
       'Taux_pauvreté_seuil_60', 'Q1', 'Mediane', 'Q3', 'Ecart_inter_Q_rapporte_a_la_mediane', 'D1', 'D2', 'D3', 'D4',
       'D5', 'D6', 'D7', 'D8', 'D9', 'Rapport_interdécile_D9/D1', 'S80/S20', 'Gini', 'Part_revenus_activite',
       'Part_salaire', 'Part_revenus_chomage', 'Part_revenus_non_salariées', 'Part_retraites', 'Part_revenus_patrimoine',
       'Part_prestations_sociales', 'Part_prestations_familiales', 'Part_minima_sociaux', 'Part_prestations_logement','Part_impôts']

@st.cache_data
def convert_gpd(data, equi=False):
    """
    Converts a pandas DataFrame to a GeoDataFrame using the geometry attribute.
    
    Args:
        : A pandas DataFrame with longitude and latitude columns.
        equi: A boolean flag indicating whether the speicified DataFrame is 'equipements'.

    Returns:
        A GeoDataFrame with a 'geometry' column containing points corresponding to the latitude and 
        longitude or Lambert coordinates of the input DataFrame.
    
    Raises:
        ValueError: If the input DataFrame does not contain the expected columns.
    """
    try:
        if equi:
            return gpd.GeoDataFrame(
                data, geometry = gpd.points_from_xy(data.LAMBERT_X, data.LAMBERT_Y)
            )
        return gpd.GeoDataFrame(
                data, geometry = gpd.points_from_xy(data['longitude'], data['latitude'])
            )
    except ValueError as e:
        print(f"Error converting to GeoDataFrame: {e}")
        return None


@st.cache_data
def iris_prep(iris_value, iris_shape):
    """
    Merge iris_shape and iris_value tables to obtain the polygons and the IRIS values in the same table.

    Args: None
    Returns: A pandas dataframe containing the merged iris data with no duplicate entries based on 
    'DCOMIRIS' column.
    """
    try:
        # Remove duplicates from iris_shape and iris_value tables
        iris_shape = iris_shape.drop_duplicates(subset=['DCOMIRIS'], keep='first')
        iris_value = iris_value.drop_duplicates(subset=['IRIS'], keep='first')

        # Convert 'IRIS' column to a string of 9 characters with leading zeros if necessary
        iris_value['IRIS'] = iris_value['IRIS'].astype(str).str.rjust(9, '0')

        # Merge iris_shape and iris_value tables and remove duplicates based on 'DCOMIRIS' column
        iris = iris_shape.merge(iris_value, how='left', right_on='IRIS', left_on='DCOMIRIS')
        iris = iris.drop_duplicates(subset=['DCOMIRIS'], keep='first')

        return iris
    except KeyError as e:
        print(f"Error: {str(e)} column not found in input data")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

@st.cache_data
def equipements_prep(liste_iris):
    """
    Aggregate the number of equipment for selected categories at the IRIS level.

    Args:
        liste_iris (list): list of IRIS to include in the aggregation.

    Returns:
        pd.DataFrame: dataframe containing the aggregated number of equipment for the selected 
        categories at the IRIS level.
    
    Raises:
        ValueError: If the input list of IRIS is empty.
    """
    
    print("Adding amenities...")
    
    # Read amenities file
 
    global amenities
    # Filter the amenities dataframe to only include IRIS of interest
    amenities = amenities[amenities['DCIRIS'].isin(liste_iris)]
    amenities_df = []

    for equipement in liste_equipements:
        # Filter the amenities dataframe to only include the current equipment category
        amenities_temp = amenities[amenities['TYPEQU'].isin(equipement)]

        # Group the amenities dataframe by DCIRIS and TYPEQU, count the number of occurrences and 
        #store the result in a dataframe
        amenities_temp = amenities_temp.groupby('DCIRIS')['TYPEQU'].value_counts().to_frame()

        # Group the amenities dataframe by DCIRIS, sum the number of equipment and rename the 
        #column to the first equipment name in the list
        amenities_temp = amenities_temp.groupby('DCIRIS').sum()
        amenities_temp = amenities_temp.rename(columns={"TYPEQU": equipement[0]})

        # Append the amenities dataframe to the amenities list
        amenities_df.append(amenities_temp)

    # Concatenate the amenities dataframes in the amenities list, fill the missing values with 0, 
    #and reset the index
    amenities_df = pd.concat(amenities_df).fillna(0)
    amenities_df['DCIRIS'] = amenities_df.index
    amenities_df = amenities_df.reset_index(drop=True)
    
    # drop duplicates and group by IRIS
    amenities_df = amenities_df.drop_duplicates()
    amenities_df = amenities_df.groupby(["DCIRIS"], as_index=False).sum()

    return amenities_df

@st.cache_data
def prep_lyc(data: pd.DataFrame, geo_etab: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Filters the given lycée data to only include lycées généraux, as they are more likely to
    influence housing prices than other types of schools. Calculates the taux de mention for
    each lycée and converts the result to a geopandas dataframe, which is then merged with
    the dvf data.

    Args:
        data (pd.DataFrame): a pandas DataFrame containing data on lycées
        geo_etab (pd.DataFrame): a pandas DataFrame containing geographical data on the lycées

    Returns:
        A geopandas GeoDataFrame with the filtered and processed lycée data
    """    
    try:
        # Start by filtering out the data for years other than 2020 and keeping only lycées généraux
        lyc = data[data['Annee'] == 2020]
        lyc_gen = lyc[['Etablissement', 'UAI', 'Code commune',
                    'Presents - L', 'Presents - ES', 'Presents - S',
                    'Taux de mentions - L', 
                    'Taux de mentions - ES',
                    'Taux de mentions - S']]
        lyc_gen = lyc_gen[(lyc_gen['Presents - L']>0) |
            (lyc_gen['Presents - ES']>0)|
            (lyc_gen['Presents - S']>0)]
        lyc_gen = lyc_gen.fillna(0)
        # Calculate the taux de mention for each lycée
        lyc_gen['taux_mention'] = (lyc_gen['Presents - L'] * lyc_gen['Taux de mentions - L'] + lyc_gen['Presents - ES'] * lyc_gen['Taux de mentions - ES'] + lyc_gen['Presents - S'] * lyc_gen['Taux de mentions - S']) / (lyc_gen['Presents - S'] + lyc_gen['Presents - L'] + lyc_gen['Presents - ES'])
        # Merge the lycée data with the geographical data
        lyc_gen = lyc_gen.merge(geo_etab, how = 'left', left_on = 'UAI', right_on = 'numero_uai')
        # Select only the relevant columns and rename them for clarity
        lyc_gen = lyc_gen[['Etablissement', 'UAI', 'Code commune', 'code_departement',
                'Taux de mentions - L', 'Taux de mentions - ES', 'Taux de mentions - S', 'taux_mention',
                'latitude', 'longitude']]
        lyc_gen.rename(columns = {'Taux de mentions - L':'taux_mention_L', 'Taux de mentions - ES':'taux_mention_ES', 'Taux de mentions - S':'taux_mention_S'}, inplace=True)
        # Convert the resulting DataFrame to a geopandas GeoDataFrame, and filter out any rows with missing geographic data
        lyc_gen_geo = gpd.GeoDataFrame(
            lyc_gen, geometry = gpd.points_from_xy(lyc_gen.longitude, lyc_gen.latitude))
        lyc_gen_geo = lyc_gen_geo[(lyc_gen_geo['latitude'].notna()) & (lyc_gen_geo['longitude'].notna())]

        return lyc_gen_geo
    except Exception as e:
        print(f"An error occurred while preprocessing lycees data: {e}")
        return None

@st.cache_data
def prep_brevet(data: pd.DataFrame, geo_etab: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Preprocesses brevet data by computing the taux de mention for each college,
    converting it to a geopandas dataframe, and merging it with the DVF dataframe.

    Args:
        data (pd.DataFrame): a pandas DataFrame containing data on lycées
        geo_etab (pd.DataFrame): a pandas DataFrame containing geographical data on the lycées

    Returns:
        A geopandas GeoDataFrame with the filtered and processed lycée data
    """
    try:
        brevet = data[data['session'] == 2021]
        brevet_geo = brevet.merge(geo_etab, how = 'left', left_on = 'numero_d_etablissement', right_on = 'numero_uai')
        brevet_geo = brevet_geo[['numero_uai', 'code_commune',
                                'nombre_total_d_admis', 'nombre_d_admis_mention_tb','taux_de_reussite',
                                'latitude', 'longitude']]
        brevet_geo['taux_mention'] = brevet_geo['nombre_d_admis_mention_tb'] / brevet_geo['nombre_total_d_admis']

        brevet_geo = gpd.GeoDataFrame(
            brevet_geo, geometry = gpd.points_from_xy(brevet_geo.longitude, brevet_geo.latitude))
        brevet_geo = brevet_geo[(brevet_geo['latitude'].notna()) & (brevet_geo['longitude'].notna())]

        return brevet_geo

    except TypeError as e:
        print(f"TypeError occurred while preprocessing brevet data: {e}")
        return None
    
    except KeyError as e:
        print(f"KeyError occurred while preprocessing brevet data: {e}")

@st.cache_data
def get_k_nearest_neighbors(source_points, candidate_points, k_neighbors):
    """
    Find the k nearest neighbors for all source points from a set of candidate points.
    
    Args:
        source_points: numpy array or list of arrays containing the coordinates of the source points
        candidate_points: numpy array or list of arrays containing the coordinates of the candidate points
        k_neighbors: integer specifying the number of nearest neighbors to return
    
    Returns:
        tuple containing two numpy arrays:
        - indices: the indices of the k nearest neighbors in the candidate_points array for each source point
        - distances: the distances between each source point and its k nearest neighbors
    """
    try:
        tree = BallTree(candidate_points, leaf_size=15, metric='haversine')
        distances, indices = tree.query(source_points, k=k_neighbors)
        return indices, distances
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

@st.cache_data
def get_nearest_neighbors(left_gdf, right_gdf, k_neighbors, return_distances=False):
    """
    For each point in left_gdf, find the k-nearest neighbors in right_gdf and return their indices.
    Assumes that the input Points are in WGS84 projection (lat/lon).
    """
    left_geom_col_name = left_gdf.geometry.name
    right_geom_col_name = right_gdf.geometry.name

    #ensure that index in right gdf is formed of sequential numbers
    right_gdf = right_gdf.reset_index(drop=True)

    # convert coordinates to radians
    left_radians_x = left_gdf[left_geom_col_name].x.apply(lambda geom: geom * np.pi / 180)
    left_radians_y = left_gdf[left_geom_col_name].y.apply(lambda geom: geom * np.pi / 180)
    left_radians = np.c_[left_radians_x, left_radians_y]

    right_radians_x = right_gdf[right_geom_col_name].x.apply(lambda geom: geom * np.pi / 180)
    right_radians_y = right_gdf[right_geom_col_name].y.apply(lambda geom: geom * np.pi / 180)
    right_radians = np.c_[right_radians_x, right_radians_y]

    indices, distances = get_k_nearest_neighbors(source_points=left_radians,
                                                 candidate_points=right_radians,
                                                 k_neighbors=k_neighbors)
    if return_distances:
        return indices, distances
    else:
        return indices

@st.cache_data
def apply_linear_regression(row,table_info, metric_of_interest):

    """Apply linear regression to calculate the intercept of a row with the given metric of interest."""
    indices = row['indices']
    X = table_info.loc[indices, ['surface_reelle_bati', 'nombre_pieces_principales']].values
    y = table_info.loc[indices, metric_of_interest].values

    lr = LinearRegression()
    lr.fit(X, y)

    return lr.intercept_

@st.cache_data
def calculate_closest_metric(dvf, table_info, k_neighbors, metric_of_interest, new_metric_name, apply_regression=False):
    """Compute the new metric based on the k-nearest neighbors in table_info dataframe."""
    try:
        print(f"Computing `{new_metric_name}`...")
        dvf[new_metric_name] = np.nan
        closest_indices = get_nearest_neighbors(left_gdf=dvf, right_gdf=table_info, k_neighbors=k_neighbors)
        dvf['indices'] = list(closest_indices)

        if apply_regression: 
            dvf[new_metric_name] = dvf.swifter.apply(lambda row: apply_linear_regression(row, metric_of_interest), axis=1)
        else:
            dvf[new_metric_name] = dvf['indices'].apply(lambda indices: table_info[metric_of_interest].iloc[indices].mean())

        return dvf

    except Exception as e:
        print("Error: could not calculate closest metric")
        print(str(e))
        return None

@st.cache_data
def alter_metric_name(df,input_variable_names,output_variable_names):
    """
    Calculate new metrics using my_choose_closest() function and return updated dataframe.

    Args:
        df (pandas dataframe): dataframe to calculate new metrics on.
        input_variable_names (list): names of variables to calculate new metrics from.
        output_variable_names (list): names to give new metrics.

    Returns:
        df (pandas dataframe): updated dataframe with input variables dropped and new metrics added.
    """
    # Define a helper function to calculate a single new metric using my_choose_closest() function
    def calculate_single_metric(params):
        input_var = params[0]
        output_var = params[1]
        return calculate_closest_metric(dvf=df,
                                        table_info=df[df[input_var].notnull()], 
                                        k_neighbors=1,
                                        metric_of_interest=input_var, 
                                        new_metric_name=output_var)[output_var]
    
    # Calculate the new metrics using parallel processing
    results = Parallel(n_jobs=-1, verbose=1)(
        delayed(calculate_single_metric)(vars) for vars in zip(input_variable_names, output_variable_names))
    
    # Create a dictionary of new metric names and values
    new_metrics_dict = dict(zip(output_variable_names, results))
    
    # Add the new metrics to the df dataframe
    df = df.assign(**new_metrics_dict)
    
    # Drop the input variables from the df dataframe
    df = df.drop(columns=input_variable_names)
    
    return df



income_input_variable_names = ['DISP_TP6019', 'DISP_Q119', 'DISP_MED19', 'DISP_Q319', 'DISP_EQ19', 'DISP_D119', 'DISP_D219',
                        'DISP_D319', 'DISP_D419', 'DISP_D619', 'DISP_D719', 'DISP_D819', 'DISP_D919', 'DISP_RD19',
                        'DISP_S80S2019', 'DISP_GI19', 'DISP_PACT19', 'DISP_PTSA19', 'DISP_PCHO19', 'DISP_PBEN19',
                        'DISP_PPEN19', 'DISP_PPAT19', 'DISP_PPSOC19', 'DISP_PPFAM19', 'DISP_PPMINI19', 'DISP_PPLOGT19',
                        'DISP_PIMPOT19', 'DISP_NOTE19']
income_output_variable_names = ['Taux_pauvreté_seuil_60', 'Q1', 'Mediane', 'Q3', 'Ecart_inter_Q_rapporte_a_la_mediane', 'D1',
                         'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'Rapport_interdécile_D9/D1', 'S80/S20', 'Gini',
                         'Part_revenus_activite', 'Part_salaire', 'Part_revenus_chomage', 'Part_revenus_non_salariées',
                         'Part_retraites', 'Part_revenus_patrimoine', 'Part_prestations_sociales',
                         'Part_prestations_familiales', 'Part_minima_sociaux', 'Part_prestations_logement', 'Part_impôts']

equi_input_variable_names=['A203', 'A206', 'B101', 'C101', 'C201', 'D201', 'E107', 'F303', 'F307', 'F313']

equi_output_variable_names = ['Banques', 'Bureaux_de_Poste', 'Commerces', 'Ecoles','Collèges_Lycées', 'Medecins','Gares', 'Cinema',
                        'Bibliotheques', 'Espaces_remarquables_et_patrimoine']            

@st.cache_data
def choose_metric_name(df, variable):
    """
    Calculates a new metric using the given input metric and name.
    
    Args:
        df: pandas DataFrame to modify
        variable: string indicating the type of metric to create. Should be either 'income' or 'equip'.

    Returns:
        A pandas DataFrame with a new column for the selected metric.
    """
    if variable == 'income':
        return alter_metric_name(df, income_input_variable_names, income_output_variable_names) 
    elif variable == 'amenity':
        return alter_metric_name(df,equi_input_variable_names, equi_output_variable_names)
    else :
        raise ValueError("Invalid variable input. Choose either 'income' or 'amenity'.")

@st.cache_data
def select_variables(dvf_geo, keep_columns = liste_var_garder):
    """
    Select variables from dvf_geo dataframe and return updated dataframe.

    Args:
        dvf_geo (pandas dataframe): dataframe to select variables from.
        keep_columns (list): list of variables to keep in the updated dataframe.

    Returns:
        dvf_geo_final (pandas dataframe): updated dataframe with selected variables.
    """
    try:
        if not isinstance(dvf_geo, pd.DataFrame):
            raise TypeError("dvf_geo must be a pandas DataFrame.")
        
        print("Keeping variables of interest...")
        # Keep columns of interest
        keep_columns=list(set(keep_columns)&set(dvf_geo.columns))
        dvf_geo_final = dvf_geo[keep_columns]
        return dvf_geo_final

    except KeyError as e:
        print(f"Error occurred while selecting variables: {e}")
        return None

    except TypeError as e:
        print(f"Error occurred while filtering data: {e}")
        return None
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    
@st.cache_data
def load_data(path, delimiter = None, header = 0, geopanda=False):
    if delimiter != None:
        if header == 0:
            return(pd.read_csv(path, delimiter = delimiter, storage_options=storage_options))
        else:
            return(pd.read_csv(path, delimiter = delimiter, storage_options=storage_options, header=header))
    else:
        return(pd.read_csv(path, storage_options=storage_options))
            

#------------------------------CHARGEMENT DES BASES DE DONNEES COMPLEMENTAIRES----------------------------------------------

import zipfile

with st.spinner("Chargement des données..."):  
    
    data = load_data('Final_csv')
    iris_value = load_data('IRIS_donnees', delimiter=';')
    iris_shape = gpd.read_file('IRIS_contours.shp')
    
    amenities = load_data('bpe21_ensemble_xy.csv', delimiter=';')
    geo_etab = load_data('geo_brevet.csv', delimiter=';')
    brevet = load_data('resultats_brevet.csv', delimiter=';')
    lyc = load_data('resultats_lycées.csv', delimiter=';')
    metropoles = load_data('metropoles_communes.csv', delimiter=';', header = 5)
    
    st.write(data)
    data_2 = load_data_from_drive('https://drive.google.com/file/d/1CgGNYXtoNHpyGFFc3eIygvu2VEIlkljX/view?usp=sharing', delimiter=';')
    st.write(data_2)
    data = convert_gpd(data)
         
#------------------------------DEMANDE DE L'ADRESSE----------------------------------------------

with st.sidebar:
    adresse = st.text_input("Veuillez entrer l'adresse:")

#------------------------------SI PAS D'ADRESSE RENTREE : PAGE DE PRESENTATION----------------------------------------------
if adresse == '':
    st.markdown("**Présentation de l'application** - Cette application a été créée dans le cadre du projet académique *Business Data Challenge* de l'ENSAE effectué en partenariat avec meilleurtaux.com 📈. Elle a été créée dans le but d'exposer le résultat de nos travaux et proposer une démonstration ludique des capacités de l'IA en matière de prédiction de prix de l'immbolier 🔮 Attention, les résultats sont affichés à titre indicatif et nous ne garantissons aucun résultat ⚠️")
    st.markdown("**Fonctionnement de l'application** - L'application permet d'utiliser notre modèle d'XGBoost permettant d'estimer les prix de biens immobiliers situés dans l'une des métropoles suivantes: Paris🗼, Marseille☀️, Lyon🦁, Lille⛏, Bordeaux🍷, Toulouse🏉, Montpellier🏖️, Nantes🔰, Rennes🦌, Nice😎. Le modèle détecte automatiquement si le modèle est compatible avec l'adresse rentrée🔄. Il faut ensuite renseigner quelques informations sur la nature du bien et le prix est calculé 🏷️.")
    st.markdown("**Pour plus d'informations** sur le fonctionnement du modèle et du traitement de la donnée, notre travail est disponible sur la forme de package. La documentation est disponible sur le GitHub : https://github.com/SalahMouslih/Data-challenge")           

#------------------------------API MAPS, MESSAGE D'ERREUR DANS LA SIDEBAR----------------------------------------------

with st.sidebar:
        
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
     if adresse != '':
        st.write('Adresse non trouvée')

    type_bien = st.selectbox("Sélectionner le type de bien",("Appartement", "Maison"))

#------------------------------AFFICHAGE DE LA MAP SI ON A BIEN UNE LAT LONG----------------------------------------------    

with col1:
    if type(lat_lon) != NoneType:
        st.map(data=lat_lon)

#------------------------------IMPORTATION DU MODELE----------------------------------------------

Bool_User_Ville_Succesful = np.isin(ville,['Paris','Marseille','Lyon','Lille','Bordeaux','Toulouse','Nice','Nantes','Montpellier','Rennes'])

if Bool_User_Ville_Succesful:

    pipe = joblib.load('{}-{}.joblib'.format(ville,type_bien))
    
    with col2:
        st.write('Modele Chargé pour la ville de {} pour un bien de type {}'.format(ville,type_bien))
    preprocessor = pipe[:-1]
    st.write(preprocessor.named_steps, preprocessor.feature_names_in_)
    xgb_model = pipe[-1]
    
else:
    with col2:
        if ville != None:
            st.write("La ville de {} n'est pas couverte par notre modèle. Veuillez réessayer dans l'une des métropoles suivantes : Paris, Marseille, Lyon, Lille, Bordeaux, Toulouse, Nice, Nantes, Montpellier, Rennes".format(ville))

#------------------------------IMPORTATION DE LA BASE DE DONNEES TEST----------------------------------------------

#url = "https://drive.google.com/file/d/1PIdlpGqh8UoFYOUZCuE9kZ2ShEQTg3q1/view?usp=sharing"
#path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
#storage_options = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36'}
#data = pd.read_csv(path, storage_options=storage_options)


if Bool_User_Ville_Succesful:
   
    data_test = pd.read_csv('data_test.csv')
    data_test = data_test[(data_test['nom_commune']==ville) & (data_test['type_local']==type_bien)]

#------------------------------INPUT DES AUTRES CARACTERISTIQUES DU BIEN----------------------------------------------

    with st.sidebar:

            nombre_pieces_principales = st.slider('Nombre de pièces principales', min_value = int(min(data['nombre_pieces_principales'])),
                                                 max_value = int(max(data['nombre_pieces_principales'])), value = int(np.mean(data['nombre_pieces_principales'])), 
                                                 step = 1)
            
            surface_reelle_bati = st.slider('Surface réelle du batiment (en mètres carrés)', min_value = float(min(data['surface_reelle_bati'])),
                                                 max_value = float(max(data['surface_reelle_bati'])), value = float(np.mean(data['surface_reelle_bati'])))
            
            nombre_lots = st.slider('Nombre de lots', min_value = 0, max_value = 2, value = int(np.mean(data['nombre_lots'])), step = 1)
            
            if type_bien == "Maison":             
                surface_terrain = st.slider('Surface du terrain de la maison (en mètres carrés)', min_value = 0,
                                                 max_value = 10000, value=500)       

    #------------------------------CREATION DE LA DONNEE ENTRANTE COMPLETE----------------------------------------------
      

    if type_bien == 'Appartement':
        surface_terrain = np.nan()
        
    test_data = pd.DataFrame({'code_departement':code_departement,
                                  'nombre_lots':nombre_lots,
                                  'type_local':type_bien,
                                  'surface_reelle_bati':surface_reelle_bati,
                                  'nombre_pieces_principales':nombre_pieces_principales,
                                  'surface_terrain':surface_terrain,
                                  'longitude':float(lat_lon['lon']),
                                  'latitude':float(lat_lon['lat'])})
          
#------------------------------FONCTION DE PREPROCESSING FINALE----------------------------------------------
    
    def prepreocessing_to_predict(test_data):
     
        # Create the variable "prix moyen au m2 des 10 biens les plus proches"
        dvf_geo = calculate_closest_metric(dvf = test_data,
                table_info = data,
                k_neighbors = 10,
                metric_of_interest = 'prix_m2_actualise',
                new_metric_name = 'prix_m2_zone')
        dvf_geo = dvf_geo.reset_index(drop=True)


        # Get the taux de mention for each lycée and collège as well as their geographical coordinates
        
        lyc_gen_geo = prep_lyc(lyc, geo_etab)
        brevet_geo = prep_brevet(brevet, geo_etab)

        # Calculate the average 'taux de mention' of the 3 closest 'lycées' for each property
        dvf_geo = calculate_closest_metric(dvf=dvf_geo, table_info=lyc_gen_geo,
                                            k_neighbors=3,
                              
                                           metric_of_interest='taux_mention',
                                            new_metric_name='moyenne')

        # Calculate the average 'taux de mention' of the 3 closest 'collèges' for each property
        dvf_geo = calculate_closest_metric(dvf=dvf_geo, table_info=brevet_geo,
                                            k_neighbors=3,
                                            metric_of_interest='taux_mention',
                                            new_metric_name='moyenne_brevet')

        # Add information about the IRIS area
        
        iris = iris_prep(iris_value, iris_shape)
        dvf_geo = dvf_geo.sjoin(iris, how = 'left', predicate = 'within')

        #Choose the metric name for income
        dvf_geo = choose_metric_name(dvf_geo,'income')


        #Add information about the equipment available in the area
        liste_iris = dvf_geo['DCOMIRIS'].unique()
        equipements = equipements_prep(liste_iris)

        dvf_geo = dvf_geo.merge(equipements, how = 'left', left_on = 'DCOMIRIS', right_on = 'DCIRIS')
        dvf_geo = choose_metric_name(dvf_geo,'amenity')
        dvf_geo=select_variables(dvf_geo)
        to_drop=['adresse_numero', 'adresse_suffixe','numero_disposition',
       'adresse_code_voie', 'code_postal', 'code_commune',
       'ancien_code_commune','ancien_nom_commune' , 'ancien_id_parcelle',
       'numero_volume', 'lot1_numero', 'lot1_surface_carrez', 'lot2_numero',
       'lot2_surface_carrez', 'lot3_numero', 'lot3_surface_carrez',
       'lot4_numero', 'lot4_surface_carrez', 'lot5_numero',
       'lot5_surface_carrez','code_type_local',
       'code_nature_culture', 'nature_culture', 'code_nature_culture_speciale',
       'nature_culture_speciale','id_mutation','id_parcelle','numero_disposition', 'nature_mutation','valeur_fonciere',
       'id_parcelle','nature_mutation','date_mutation','LIBEPCI','prix_actualise','DCOMIRIS','DCIRIS','prix_m2',
       'type_local','geometry','indices','quantile_prix','coeff_actu']
        drop_clean=list(set(dvf_geo.columns)&set(to_drop))
        dvf_geo.drop(drop_clean,axis=1,inplace=True)
        return dvf_geo

    #------------------------------PREDICTION----------------------------------------------
    
    test_result = result=prepreocessing_to_predict(test_data)
    prediction = pipe.predict(test_result)

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
