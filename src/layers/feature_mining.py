import tensorflow as tf

class CustomLayer(tf.keras.layers.Layer):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)      
        self.d1 = tf.keras.layers.Dense(512, activation=tf.nn.leaky_relu,name='dense9')
        self.d2 = tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu,name='dense10')
        self.d4 = tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu,name='dense11')
        self.out1 = tf.keras.layers.Dense(6, activation=tf.nn.softmax,name='dense12') 
        self.d5 = tf.keras.layers.Dense(512, activation=tf.nn.leaky_relu,name='dense13')
        self.d6 = tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu,name='dense14')
        self.d8 = tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu,name='dense15')
        self.out2 = tf.keras.layers.Dense(6, activation=tf.nn.softmax,name='dense16') 
        self.d9 = tf.keras.layers.Dense(512, activation=tf.nn.leaky_relu,name='dense17')
        self.d10 = tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu,name='dense18')
        self.d12 = tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu,name='dense19')
        self.out3 = tf.keras.layers.Dense(6, activation=tf.nn.softmax,name='dense20')

    def call(self, inputs):
        merged, mask_result, mask_result2 = inputs
        
        d1_output = self.d1(merged)
        d2_output = self.d2(d1_output)
        d4_output = self.d4(d2_output)
        out1_output = self.out1(d4_output)
        
        d5_output = self.d5(mask_result)
        d6_output = self.d6(d5_output)
        d8_output = self.d8(d6_output)
        out2_output = self.out2(d8_output)
        
        d9_output = self.d9(mask_result2)
        d10_output = self.d10(d9_output)
        d12_output = self.d12(d10_output)
        out3_output = self.out3(d12_output)
        
        return out1_output, out2_output, out3_output
