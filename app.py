## Archivo Pythnon para probar el STREAMLIT

# Ejecutar dede consola: conda activate envjtx2023
# instalar el streamlit: pip install streamlit
# ubiación archivo: C:\000D\Archix\Python\000Vstudio\Streamlit\StreamlitPrueba02
# muevete a la ubicación del archivo en la consola MSDOS: cd C:\000D\Archix\Python\000Vstudio\Streamlit\StreamlitPrueba02
# (envjtx2023) C:\000D\Archix\Python\000Vstudio\Streamlit\StreamlitPrueba01>streamlit run app.py
# con esto abre 
#  Local URL: http://localhost:8501
#  Network URL: http://192.168.1.181:8501
# luego según voy modificando el APP.py actualizo pulsando R en el navegador
# NOTA: si ya tuviera en uso el localhost:8501 usaría el 8502
# NOTA: Guarda archivo .py y luego Pulsando R en la web, se actualiza 

# NOTA: deploy en streamlit.io
# tienes que crear archivo requirements.txt en GitHub con las librerias a instalar.
# y poner todo en GitHub


## PASOS DEL PROYECTO
# Primero: decidir el diseño:
#


### LIBRERIAS
import streamlit as st  #libreria de streamlit
import pandas as pd 

#ML
#import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score




### Pagina web
## Containers
header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()


### Formato de la pagina
st.markdown(
    '''
    <style>
    .main{
    background-color: #F5F5F5;
    }
    </style>
    ''',
    unsafe_allow_html=True
)



### funciones
#@st.cache    # ESTO NO HACE FALTA (si para grandes tablas de datos), pero así, deja los datos en cache y no los tiene que leer cada vez que modificas algo en la pagina
    #NOTA: st.cache is deprecated. 
def get_data(filename):
    taxi_data = pd.read_csv(filename)
    return taxi_data


### Pagina web
# container 01 cabecera
with header:
    st.title("Bienvenido a mi Web Page de un proyecto de Data Science")
    #st.text("En este ejemplo voy a ver el dinero de las transiciones de los taxistas de NYC")
    #st.write("En este ejemplo voy a ver el dinero de las transiciones de los taxistas de NYC")
    st.markdown("En este ejemplo voy a ver el dinero de las transiciones de los taxistas de NYC")


# container 02 Dataset
with dataset:
    st.header("NYC Taxi Dataset (en realidad uso IRIS.CSV")
    st.text("Dataset descargado de esta web https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page")

    #taxi_data = pd.read_csv("data/taxi_data.csv")
    #taxi_data = pd.read_csv("data/iris.csv")
    taxi_data = get_data("data/iris.csv")   #uso la funcion que cree arriba
    st.write(taxi_data.head())

    st.subheader('Pick up location ID distribution on the NYC Dataset')
    #pulocation_dist = pd.DataFrame(taxi_data['PULocationID'].value_counts())
    pulocation_dist = pd.DataFrame(taxi_data['petal.width'].value_counts()).head(10)
    #pulocation_dist = pd.DataFrame(taxi_data['variety'].value_counts()).head(10)
    st.bar_chart(pulocation_dist)

# container 03 Features
with features:
    st.header("Features que he creado")

    st.markdown("* **first feature:** I created this feature beacues of this... I calculated it usin...")
    st.markdown("* **second feature:** I created this feature beacues of this... I calculated it usin...")
    st.markdown("* **third feature:** I created this feature beacues of this... I calculated it usin...")
    
    


# container 04 modeltraining
with model_training:
    st.header("Tiempo de entrenar el modelo")
    st.text("Aqui escogemos los hiperparámetros del modelo y vemos como afectan los cambios")

    # Poner columnas dentro del container
    sel_col, disp_col = st.columns(2)   # 2 columnas, una se llama sel_Col y la otra disp_col

    #slider
    max_depth = sel_col.slider('Cual debería ser la max_depth del modleo?', min_value=10, max_value=100, value=20, step=10)
    #selectbox
    n_estimators = sel_col.selectbox("cuantos arboles deberia haber?", options=[100,200,300,"No limit"], index=0)


    #lista de features
    sel_col.text("Aqui hay una lista de features demis datos:")
    sel_col.write(taxi_data.columns)
    
    #Text imput
    #input_feature = sel_col.text_input("Que feature debería usarse como input feature?","PULocationID")
    input_feature = sel_col.text_input("Que feature debería usarse como input feature?","petal.width")
    #input_feature = sel_col.text_input("Que feature debería usarse como input feature?","variety")



    #MAchine LEarning
    if n_estimators == "No limit":
        regr = RandomForestRegressor(max_depth=max_depth)
    else:
        regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)


    x = taxi_data[[input_feature]]
    #y = taxi_data[['trip_distance']]
    y = taxi_data[['petal.width']]

    regr.fit(x,y)
    prediction = regr.predict(y)

    disp_col.subheader('Mean absolute error of the model is:')
    disp_col.write(mean_absolute_error(y, prediction))

    disp_col.subheader('Mean squared  error error of the model is:')
    disp_col.write(mean_squared_error(y, prediction))

    disp_col.subheader('R squared score of the model:')
    disp_col.write(r2_score(y, prediction))
