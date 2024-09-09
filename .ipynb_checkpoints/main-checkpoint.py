import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow import keras

model = tf.keras.models.load_model('94.69_accuracy_model.keras')
print("model:", model)

st.title('Credit Risk Prediction App')