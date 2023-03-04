import tensorflow as tf
import math

class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = tf.keras.layers.Dense(512, name='dense1')
        self.dense2 = tf.keras.layers.Dense(300, name='dense2')
        self.dense3 = tf.keras.layers.Dense(1, activation=tf.nn.tanh, name='dense3')
        self.dense4 = tf.keras.layers.Dense(1, activation=tf.nn.tanh, name='dense4')
        self.dense5 = tf.keras.layers.Dense(512, name='dense5')
        self.dense6 = tf.keras.layers.Dense(300, name='dense6')
        self.dense7 = tf.keras.layers.Dense(512, name='dense7')
        self.dense8 = tf.keras.layers.Dense(300, name='dense8')
        self.add = tf.keras.layers.Add()
        self.transpose = tf.keras.layers.Lambda(lambda x: tf.transpose(x))

    def call(self, inputs):
        visual, signal = inputs
        self_visual = self.dense1(visual)
        self_impulse = self.dense2(signal)
        cc_visual = tf.matmul(self.transpose(self_visual), self_impulse)
        cc_impulse = tf.matmul(self.transpose(self_impulse), self_visual)
        visual_att = self.dense3(tf.math.divide(cc_visual, math.sqrt(300)))
        impulse_att = self.dense4(tf.math.divide(cc_impulse, math.sqrt(512)))
        cross_visual_impulse = self.dense5(self.transpose(visual_att))
        cross_impulse_visual = self.dense6(self.transpose(impulse_att))
        H_visual_att = self.dense7(cross_visual_impulse)
        H_impulse_att = self.dense8(cross_impulse_visual)
        att_visual_features = self.add([H_visual_att, visual])
        att_impulse_features = self.add([H_impulse_att, signal])
        return att_visual_features, att_impulse_features