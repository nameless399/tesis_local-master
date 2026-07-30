import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # evita el segfault
from keras.models import load_model

model = load_model("models_mix2/mix_cnn_lstm_T32_F51.keras", 
                   compile=False)
model.summary()