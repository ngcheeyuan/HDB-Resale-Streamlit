import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import folium
from streamlit_folium import st_folium

def haversine(lat1,lon1,lat2,lon2):
    lat1 , lon1 , lat2 , lon2 = map(np.deg2rad,\
                                     [lat1,lon1,lat2,lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*\
        np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * 6371* np.arcsin(np.sqrt(a))
    
    return c

def map_nearest_transactions(block,town,year):
    town = town.upper()
    coordinates = getcoordinates(str(block) +' '+town)
    
    mask1 = df.town == town
    mask2 = df.year_transacted >= year
    temp_df = df[mask1&mask2].copy()
    temp_df['distance'] = haversine(temp_df.lat.values,temp_df.lon.values,coordinates[0],coordinates[1])
    mask3 = temp_df['distance'] < 0.5
    temp_df1 = temp_df[mask3]

    mymap = folium.Map(location = [coordinates[0], coordinates[1]], 
                   width = 950, 
                   height = 550,
                   zoom_start = 18, 
                   tiles = 'openstreetmap')
    
    
    for row in temp_df1[['block','street_name','lat','lon','resale_price','month','storey_range','estimated_lease','flat_type','flat_model']].iterrows():
        block,street_name,lat,lon,resale_price,month,storey_range,estimated_lease,flat_type,flat_model = row[1]
        iframe = folium.IFrame(f'Address: BLK {block} {street_name} <br>\
                                 Storey: {storey_range} <br>\
                                 Flat type: {flat_type} <br>\
                                 Flat Model: {flat_model} <br>\
                                 Remaining lease: {int(estimated_lease)} yrs <br>\
                                 Price: ${int(resale_price)} <br>\
                                 Sell date: {month}')
        
        if resale_price > 600_000:
            color = 'red'
        elif (resale_price >400_000) & (resale_price <=600000):
            color = 'orange'
        else:
            color = 'green'
        
        popup = folium.Popup(iframe, min_width=300, max_width=300)
        neighbour = folium.Marker(
                location=[lat, lon],
                popup = popup,
                icon=folium.Icon(color=color)
            )   
        neighbour.add_to(mymap)
    return mymap , temp_df1

import time
def getcoordinates(address):
    time.sleep(61/250) #to satisfy API querying limit
    req = requests.get(f'https://developers.onemap.sg/commonapi/search?searchVal={address}&returnGeom=Y&getAddrDetails=N&pageNum=1')
    resultsdict = eval(req.text)
    if len(resultsdict['results'])>0:
        return float(resultsdict['results'][0]['LATITUDE']), float(resultsdict['results'][0]['LONGITUDE'])
    else:
        pass

import requests

# Force IPv4, run this before any requests
import socket
import requests.packages.urllib3.util.connection as urllib3_cn
 
def allowed_gai_family():
    family = socket.AF_INET    # force IPv4
    return family
 
urllib3_cn.allowed_gai_family = allowed_gai_family

def plot_graphs(df):

    fig = plt.figure(figsize=(16, 9))
    ax1 = plt.subplot(2,3,1)
    ax2 = plt.subplot(2,3,2)
    ax3 = plt.subplot(2,3,3)
    ax4 = plt.subplot(2,1,2)
    axes = [ax1, ax2, ax3, ax4]

    sns.boxplot(x = df.estimated_remaining_lease , ax = ax1)
    sns.histplot(x = df.estimated_remaining_lease , binwidth= 5 , ax = ax2)
    sns.boxplot(y = 'resale_price' , x ='flat_type', data = df , ax = ax3)
    sns.swarmplot(y = 'resale_price' , x ='flat_type', data = df , alpha = 0.5, ax = ax3)
    sns.scatterplot(data = df , y = 'resale_price' , x = 'month' , hue = 'flat_type')


    ax3.set_title('Resale Price distribution')
    ax1.set_title('Remaining lease distribution')
    ax2.set_title('Remaining lease distribution')

    ax4.set_ylabel('Sale Price - Excluding COV')
    ax4.set_xlabel('Sale Month')

    st.pyplot(fig)

df_1 = pd.read_csv('mapping_df1.csv')
df_2 = pd.read_csv('mapping_df2.csv')
df = pd.concat([df_1,df_2])

df['month'] = pd.to_datetime(df.month,format='%d/%m/%Y')
df['month'] = df.month.dt.date

header = st.container()
dataset = st.container()
map_1 = st.container()
graphs_2 = st.container()

with header:

    st.title("HDB Price Analysis")
    st.markdown("""The aim of this project is to analyse HDB price trends, and help buyers and sellers get a feel of the market conditions.
    Do note that the data does not include COV charges.""")

with map_1:
    st.title("Mapping recent transactions that were made around your neighbourhood")
    st.markdown("""Please input the HDB block and town information below. 
        The map generated will include all transactions made within 500 metres of the address in 2022 thus far.""")
    st.markdown("""If the flat sold for less than $400,000 , the markers will be labelled green. """)
    st.markdown("""If the flat sold for between \$400,000 and \$600,000 , the markers will be labelled orange. """)
    st.markdown("""If the flat sold for more than $600,0000 , the markers will be labelled red. """)
    addresses = list(df.town.unique())

    town = st.selectbox('Please Input a town',addresses)
    mask = df['town'] == town
    blk = st.text_input("Please Input a HDB block")
    load = st.checkbox('Load Data')
    if "load_state" not in st.session_state:
        st.session_state.load_state = False

    if load or st.session_state.load_state:
        st.session_state.load_state = True
        mymap , temp_df = map_nearest_transactions(blk,town,2022)
        st_folium(mymap)

with graphs_2:
     st.title("Summary Statistics of recent transactions in the neighbourhood made in 2022")
     st.markdown("""These graphs summarises the recent transactions that were made in 2022. \
                They only include transactions that were made within 500 metres radius of the HDB block that you have input into the cell above.""")
     if load or st.session_state.load_state:
        plot_graphs(temp_df)
        st.dataframe(temp_df.groupby('flat_type').agg({'resale_price':['min','mean','median','max','count']}).apply(pd.to_numeric, downcast='integer'))
        st.session_state.load_state = False