# Gradio - To deploy the model
import gradio as gr
import numpy as np
import joblib

# load the model and features
model = joblib.load('xgb_model.pkl')
features = joblib.load('top10_features.pkl')

# prediction function
def predict_cover(*args):
    input_data = np.array(args).reshape(1, -1)
    pred = model.predict(input_data)[0] + 1
    return f"🌲Predicted Forest Cover type {pred}"

# Build dynamic input fields 
inputs = []
for i in features:
    if "Hillshade" in i:
        inputs.append(gr.Slider(0, 255, label = i))
    elif "Distance" in i:
        inputs.append(gr.Number(label = i))
    elif i == "Aspect":
        inputs.append(gr.Slider(0, 360, label = i))
    elif i == "Slope":
        inputs.append(gr.Slider(0, 60, label = i))
    else:
        inputs.append(gr.Slider(1850, 4000, label = i))  # Elevation

# UI
gr.Interface(
    fn = predict_cover,
    inputs = inputs,
    outputs = "text", 
    title = "Forest Cover Type Prediction"
).launch()