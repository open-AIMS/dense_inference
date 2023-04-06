import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model, Model


def load_fx_model(pth):
    model = load_model(pth, compile=False)

    feature_layer_name = 'global_average_pooling2d_1' #'avg_pool'

    model = Model(inputs=model.inputs,
                  outputs=model.get_layer(feature_layer_name).output)

    return model


def index(gen, model):
    # model = load_fx_model(model_pth)

    vectors = model.predict(gen)

    return vectors

