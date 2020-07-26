from keras import backend as K
from keras.layers import Layer,InputSpec, multiply

import tensorflow as tf

class SelfAttention(Layer):
    def __init__(self, ch, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.channels = ch
        self.filters_f_g = self.channels // 8
        self.filters_h = self.channels

    def build(self, input_shape):
        kernel_shape_f_g = (1, 1) + (self.channels, self.filters_f_g)
        print(kernel_shape_f_g)
        kernel_shape_h = (1, 1) + (self.channels, self.filters_h)
        self.N = input_shape[1]*input_shape[2]
        # Create a trainable weight variable for this layer:
        self.gamma = self.add_weight(name='gamma', shape=[1], initializer='zeros', trainable=True)
        self.kernel_f = self.add_weight(shape=kernel_shape_f_g,
                                        initializer='glorot_uniform',
                                        name='kernel_f')
        self.kernel_g = self.add_weight(shape=kernel_shape_f_g,
                                        initializer='glorot_uniform',
                                        name='kernel_g')
        self.kernel_h = self.add_weight(shape=kernel_shape_h,
                                        initializer='glorot_uniform',
                                        name='kernel_h')
        self.bias_f = self.add_weight(shape=(self.filters_f_g,),
                                      initializer='zeros',
                                      name='bias_F')
        self.bias_g = self.add_weight(shape=(self.filters_f_g,),
                                      initializer='zeros',
                                      name='bias_g')
        self.bias_h = self.add_weight(shape=(self.filters_h,),
                                      initializer='zeros',
                                      name='bias_h')
        super(SelfAttention, self).build(input_shape)
        # Set input spec.
        self.input_spec = InputSpec(ndim=4,
                                    axes={3: input_shape[-1]})
        self.built = True


    def call(self, x):
        def hw_flatten(x):
            return K.reshape(x, shape=[K.shape(x)[0], K.shape(x)[1]*K.shape(x)[2], K.shape(x)[-1]])

        f = K.conv2d(x,
                     kernel=self.kernel_f,
                     strides=(1, 1), padding='same')  # [bs, h, w, c']
        f = K.bias_add(f, self.bias_f)
        g = K.conv2d(x,
                     kernel=self.kernel_g,
                     strides=(1, 1), padding='same')  # [bs, h, w, c']
        g = K.bias_add(g, self.bias_g)
        h = K.conv2d(x,
                     kernel=self.kernel_h,
                     strides=(1, 1), padding='same')  # [bs, h, w, c]
        h = K.bias_add(h, self.bias_h)

        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]
        
        beta = K.softmax(s, axis=-1)  # attention map
        
        o = K.batch_dot(beta, hw_flatten(h))  # [bs, N, C]

        o = K.reshape(o, shape=K.shape(x))  # [bs, h, w, C]
        x = self.gamma * o + x

        return [x, beta, self.gamma]

    def compute_output_shape(self, input_shape):
        return [input_shape,(self.N, self.N), (1,)]
    



class SoftAttention(Layer):
    def __init__(self,ch,m,return_x=False,aggregate=False,**kwargs):
        self.channels=int(ch)
        self.multiheads = m
        self.aggregate_channels = aggregate
        self.concat_input_with_scaled = return_x
#         self.h_w=self.channels//8
        
        super(SoftAttention,self).__init__(**kwargs)

    def build(self,input_shape):
#         kernel_shape=(1, 3, 3) + (self.channels, self.h_w)
#         print(input_shape)
        self.i_shape = input_shape
#         print('self.i_shape:',self.i_shape)
        kernel_shape_conv3d = (self.channels, 3, 3) + (1, self.multiheads) # DHWC
    
        self.out_attention_maps_shape = input_shape[0:1]+(self.multiheads,)+input_shape[1:-1]
        
        if self.aggregate_channels==False:
#             print(kernel_shape_conv3d)
            self.out_features_shape = input_shape[:-1]+(input_shape[-1]+(input_shape[-1]*self.multiheads),)
        else:
            if self.concat_input_with_scaled:
                self.out_features_shape = input_shape[:-1]+(input_shape[-1]*2,)
            else:
                self.out_features_shape = input_shape
        
#         print('self.out_features_shape:',self.out_features_shape)
        self.kernel_conv3d = self.add_weight(shape=kernel_shape_conv3d,
                                        initializer='glorot_uniform',
                                        name='kernel_conv3d')
        self.bias_conv3d = self.add_weight(shape=(self.multiheads,),
                                      initializer='zeros',
                                      name='bias_conv3d')
#         print('self.out_features_shape:',self.out_features_shape)
#         print('self.out_attention_maps_shape:',self.out_attention_maps_shape)
#         self.gamma1 = self.add_weight(name='gamma1', shape=[1], initializer='zeros', trainable=True)
#         self.gamma2 = self.add_weight(name='gamma2', shape=[1], initializer='ones', trainable=True)
        super(SoftAttention, self).build(input_shape)

    def call(self, x):
#         print('input shape:',x.shape) # None, 16,12,512
        exp_x = K.expand_dims(x,axis=-1)
#         print('expanded input shape:',exp_x.shape) # N, 16, 12, 512, 1
        # filters = Attention Heads
        c3d = K.conv3d(exp_x,
                     kernel=self.kernel_conv3d,
                     strides=(1,1,self.i_shape[-1]), padding='same', data_format='channels_last')
        conv3d = K.bias_add(c3d,
                        self.bias_conv3d)
#         print('c3d shape:', c3d.shape)
#         conv3d = kl.Conv3D(padding='same',filters=self.multiheads, kernel_size=(3,3,self.i_shape[-1]), strides=(1,1,self.i_shape[-1]),kernel_initializer='he_normal',activation='relu')(exp_x)
#         print('conv3d shape:', conv3d.shape)
        conv3d = K.permute_dimensions(conv3d,pattern=(0,4,1,2,3))
#         print('conv3d shape:', conv3d.shape)
#         conv3d = K.flatten(conv3d)
        
        conv3d = K.squeeze(conv3d, axis=-1)
        conv3d = K.reshape(conv3d,shape=(-1, self.multiheads ,self.i_shape[1]*self.i_shape[2]))
#         print('conv3d shape:', conv3d.shape)
#         conv3d = K.expand_dims(conv3d,axis=1) # N, 1, 16, 12       
#         print('conv3d shape:', conv3d.shape)
        softmax_alpha = K.softmax(conv3d, axis=-1) # attention map # N, 16x12
#         print('softmax_alpha shape:', softmax_alpha.shape)
        softmax_alpha = K.reshape(softmax_alpha,shape=(-1, self.multiheads, self.i_shape[1],self.i_shape[2]))
#         print('softmax_alpha shape:', softmax_alpha.shape)
        
        if self.aggregate_channels==False:
            exp_softmax_alpha = K.expand_dims(softmax_alpha, axis=-1) # for elementwise multiplication
    #         print('exp_softmax_alpha shape:', exp_softmax_alpha.shape)       
            exp_softmax_alpha = K.permute_dimensions(exp_softmax_alpha,pattern=(0,2,3,1,4))
    #         print('exp_softmax_alpha shape:', exp_softmax_alpha.shape)
            x_exp = K.expand_dims(x,axis=-2)
    #         print('x_exp shape:', x_exp.shape)   
            u = multiply([exp_softmax_alpha, x_exp])   
    #         print('u shape:', u.shape)   
            u = K.reshape(u, shape=(-1,self.i_shape[1],self.i_shape[2],u.shape[-1]*u.shape[-2]))
        else:
            exp_softmax_alpha = K.permute_dimensions(softmax_alpha,pattern=(0,2,3,1))
            exp_softmax_alpha = K.sum(exp_softmax_alpha,axis=-1)
#             print('exp_softmax_alpha shape:', exp_softmax_alpha.shape)
            exp_softmax_alpha = K.expand_dims(exp_softmax_alpha, axis=-1)
#             print('exp_softmax_alpha shape:', exp_softmax_alpha.shape)
            u = multiply([exp_softmax_alpha, x])   
#             print('u shape:', u.shape) 
        if self.concat_input_with_scaled:
            o = K.concatenate([u,x],axis=-1)
        else:
            o = u
#         print('o shape:', o.shape)  
#         u = kl.Conv2D(activation='relu',filters=self.i_shape[-1],kernel_size=(1,1),padding='valid')(u)
#         u = self.gamma2 * u + x
#         print('u shape:', u.shape)
#         u = self.gamma2 * u
#         u = K.tanh(u)
#         print('u shape:', u.shape)
#         return [u, softmax_alpha]
#         self.out_features_shape = tuple(u.shape.as_list())
#         self.out_attention_maps_shape = tuple(softmax_alpha.shape.as_list())
#         print(self.out_features_shape, self.out_attention_maps_shape)
        
        return [o, softmax_alpha]

    def compute_output_shape(self, input_shape): 
        return [self.out_features_shape, self.out_attention_maps_shape]
#         return [input_shape,(None,16,12),(None,1)]
    
    def get_config(self):
        return super(SoftAttention,self).get_config()