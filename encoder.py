
import tensorflow as tf

class Encoder(object):
    def __init__(self, sess, encoder, z_dim, name='Encoder'):
        self.encoder = encoder
        self.z_dim = z_dim
        self.name = name
        self.sess = sess

    def init_model(self):
        