# -*- coding: utf-8 -*-

import pandas as pd
from pycaret.regression import load_model, predict_model
from fastapi import FastAPI
import uvicorn
from pydantic import create_model
import pickle

# Create the app
app = FastAPI()

# Load trained Pipeline
# model = load_model("best_model_Extreme_Gradient_Boosting.pkl")

with open("best_model_Extreme_Gradient_Boosting.pkl", 'rb') as f:
    model = pickle.load(f)

# Create input/output pydantic models
input_model = create_model("best_model_Extreme_Gradient_Boosting_input", **{'cilindraje':1.8,
'traccion':4,
'anno':2010,
'kilometraje':98000,
'color':10,
'Paridad_de_la_placa':2,
'categoria_vehiculo_estandar':False,
'categoria_vehiculo_lujo':True,
'categoria_vehiculo_normal':False,
'categoria_vehiculo_premium':False,
'marca_Audi':False,
'marca_BMW':False,
'marca_BYD':False,
'marca_Chevrolet':False,
'marca_Citroën':False,
'marca_Dodge':False,
'marca_Fiat':False,
'marca_Ford':False,
'marca_Honda':False,
'marca_Hyundai':False,
'marca_Jeep':False,
'marca_Kia':False,
'marca_Land Rover':False,
'marca_Lexus':False,
'marca_MINI':False,
'marca_Mazda':False,
'marca_Mercedes-Benz':True,
'marca_Mitsubishi':False,
'marca_Nissan':False,
'marca_Peugeot':False,
'marca_Porsche':False,
'marca_Renault':False,
'marca_SEAT':False,
'marca_Ssangyong':False,
'marca_Subaru':False,
'marca_Suzuki':False,
'marca_Toyota':False,
'marca_Volkswagen':False,
'marca_Volvo':False,
'Tipo_de_combustible_Diésel':False,
'Tipo_de_combustible_Eléctrico':False,
'Tipo_de_combustible_Gasolina':True,
'Tipo_de_combustible_Gasolina y gas':False,
'Tipo_de_combustible_Híbrido':False,
'Carroceria_Camion':False,
'Carroceria_Camioneta':False,
'Carroceria_Convertible':False,
'Carroceria_Coupé':False,
'Carroceria_Furgón':False,
'Carroceria_Hatchback':False,
'Carroceria_Minivan':False,
'Carroceria_Pick-Up':False,
'Carroceria_Roadster':False,
'Carroceria_SUV':False,
'Carroceria_Sedán':True,
'Carroceria_Station Wagon':False,
'Carroceria_Van':False,
'Carroceria_WAGON':False,
'Carroceria_camion':False,
'Carroceria_furgón':False,
'Carroceria_minivan':False,
'transmision_Automática':True,
'transmision_Mecánica':False,
'ciudad_Abejorral':False,
'ciudad_Antonio Nariño':False,
'ciudad_Apartadó':False,
'ciudad_Armenia':False,
'ciudad_Barrios Unidos':False,
'ciudad_Bello':False,
'ciudad_Bogota':False,
'ciudad_Bosa':False,
'ciudad_Bucaramanga':False,
'ciudad_Cajicá':False,
'ciudad_Caldas':False,
'ciudad_Chapinero':False,
'ciudad_Chiquinquirá':False,
'ciudad_Chía':False,
'ciudad_Ciudad Bolívar':False,
'ciudad_Copacabana':False,
'ciudad_Cota':False,
'ciudad_Engativa':False,
'ciudad_Envigado':False,
'ciudad_Facatativá':False,
'ciudad_Fontibón':False,
'ciudad_Funza':False,
'ciudad_Girardota':False,
'ciudad_Guarne':False,
'ciudad_Itaguí':False,
'ciudad_Kennedy':False,
'ciudad_La Calera':False,
'ciudad_La Ceja':False,
'ciudad_La Estrella':False,
'ciudad_Marinilla':False,
'ciudad_Martires':False,
'ciudad_Medellín':False,
'ciudad_Mosquera':False,
'ciudad_Puente Aranda':False,
'ciudad_Rafael Uribe Uribe':False,
'ciudad_Retiro':False,
'ciudad_Rionegro':False,
'ciudad_Sabaneta':False,
'ciudad_San Cristobal Sur':False,
'ciudad_Santa Fe':False,
'ciudad_Soacha':False,
'ciudad_Suba':False,
'ciudad_Tabio':False,
'ciudad_Teusaquillo':False,
'ciudad_Tunja':False,
'ciudad_Tunjuelito':False,
'ciudad_Usaquén':True,
'ciudad_Usme':False,
'ciudad_Zipaquirá':False})
output_model = create_model("best_model_Extreme_Gradient_Boosting_output", prediction=42000000)


# Define predict function
@app.post("/predict", response_model=output_model)
def predict(data: input_model):
    data = pd.DataFrame([data.dict()])
    predictions =  model.predict(data)
    # predictions = predict_model(model, data=data)
    # return {"prediction": predictions["prediction_label"].iloc[0]}
    return {"prediction": predictions}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
