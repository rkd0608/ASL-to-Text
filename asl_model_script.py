from tensorflow.keras.models import model_from_json
from tensorflow.python.keras.backend import set_session
import numpy as np

import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
session = tf.compat.v1.Session(config=config)
set_session(session)


class ASLModel(object):

    ALPHA_LIST = ['A', 'B', 'C', 'D' , 'E' , 'F', 'G', 'H' , 'I',
						'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
						'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'space', 'nothing', 'del']

    def __init__(self, asl_model_json_file, model_weights_file):
        # load model from JSON file
        with open(asl_model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        #self.loaded_model.compile()
        #self.loaded_model._make_predict_function()

    def predict_letter(self, img):
        global session
        set_session(session)
        self.preds = self.loaded_model.predict(img)
        return ASLModel.ALPHA_LIST[np.argmax(self.preds)]
