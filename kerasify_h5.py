from tensorflow.keras.models import load_model
from kerasify import export_model

model = load_model('tiny-yolo.h5')

export_model(model, 'output.model')
