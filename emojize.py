import tensorflow as tf
from tensorflow import keras
from keras.regularizers import L1L2
from keras.layers import Input, Bidirectional, Embedding, Dense, Dropout, SpatialDropout1D, LSTM, Activation
import os 
os.sys.path.append('/home/mukulnarwani/code/deepmoji/deepmoji')
from attlayer import AttentionWeightedAverage

MAX_LEN = 4
VOCAB_SIZE = 100

def make_model():
    embeddings_L2 = 1E-6
    embeddings_regularization = L1L2(l2 = embeddings_L2)
    embeddings_output = 256
    LSTM_SIZE = 512
    NB_CLASSES= 64

    model_input = Input(shape = MAX_LEN, dtype = 'int32')
    embed = Embedding(input_dim = VOCAB_SIZE, 
                      output_dim = embeddings_output,
                      mask_zero = True,
                      input_length = MAX_LEN,
                      embeddings_regularizer = embeddings_regularization,
                      name = 'embedding')
    x = embed(model_input)
    x = Activation('tanh')(x)
    
    #LSTM layers
    lstm_0 = Bidirectional(LSTM(LSTM_SIZE, return_sequences = True), name = "bi_lstm_0")(x)
    lstm_1 = Bidirectional(LSTM(LSTM_SIZE, return_sequences = True), name = "bi_lstm_1")(lstm_0)
    #Name this better? name x skipconnections
    x = keras.layers.concatenate([lstm_1,lstm_0,x])
    x = AttentionWeightedAverage(name='attlayer')(x)
    
    outputs = [Dense(NB_CLASSES, activation='softmax', name='softmax_out')(x)]
    
    return keras.models.Model(inputs=[model_input], outputs=outputs, name="DeepMoji")
    
bertha_emoji= make_model()

bertha_emoji

