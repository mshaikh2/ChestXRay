import keras.backend as K
import keras.layers as kl
from keras.layers import Layer,InputSpec,Conv1D 
from keras import regularizers, constraints, initializers, activations
# from keras.layers.recurrent import Recurrent
import tensorflow as tf
import numpy as np
    
activation='relu'

class Attention(Layer):
    def __init__(self, ch, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.channels = ch
        self.filters_q_k = self.channels//8       
        self.filters_v = self.channels
        

    def build(self, input_shape):
        super(Attention, self).build(input_shape)
#         self.input_spec = InputSpec(shape=[(None,None,None),(None,None,None)])
#         self.built = True
        
    def call(self, inputs):
        h,x = inputs # context_vector, multi_timestep input
#         print('attn: h.shape:{0},x.shape{1}'.format(h.shape,x.shape))
        if len(h.shape)<3:
            h = K.expand_dims(axis=1,x=h)

#         print(x)
        q = kl.Dense(self.filters_q_k,use_bias=True)(h)
        q = kl.Activation(activation)(q)
#         print('attn: q.shape:{0}'.format(q.shape))
        
        k = kl.Dense(self.filters_q_k,use_bias=True)(x)
        k = kl.Activation(activation)(k)
        
        v = kl.Dense(self.filters_v,use_bias=True)(x)
        v = kl.Activation(activation)(v)
#         v = kl.tanh(alpha=1.0)(v)
#         print('q.shape,k.shape,v.shape,',q.shape,k.shape,v.shape)
        s = K.batch_dot(q, K.permute_dimensions(k,(0,2,1)))  # # [bs, 1, N]
    
#         print('s.shape:',s.shape)
        beta = K.softmax(s, axis=-1)  # attention map (bs, 1, N)
        self.beta_shape = tuple(beta.shape[1:].as_list())
#         print('beta.shape:',beta.shape)
        
        o = K.batch_dot(beta, v) 
        self.out_shape = tuple(o.shape.as_list())
        return [o, s]

    def compute_output_shape(self, input_shape):
        return [self.out_shape, self.beta_shape]

class SelfAttention(Layer):
    def __init__(self, ch, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.channels = ch
        self.filters_q_k = self.channels // 8
        self.filters_v = self.channels
        
    def build(self, input_shape):
#         kernel_shape_q_k = (1, self.channels, self.filters_q_k)
#         kernel_shape_v = (1, self.channels, self.filters_v)
#         self.N = input_shape[1]        
# #         self.gamma = self.add_weight(name='sa_gamma', shape=[1], initializer='zeros', trainable=True)
#         self.kernel_q = self.add_weight(shape=kernel_shape_q_k,
#                                         initializer='glorot_uniform',
#                                         name='kernel_q', trainable=True)
#         self.kernel_k = self.add_weight(shape=kernel_shape_q_k,
#                                         initializer='glorot_uniform',
#                                         name='kernel_k', trainable=True)
#         self.kernel_v = self.add_weight(shape=kernel_shape_v,
#                                         initializer='glorot_uniform',
#                                         name='kernel_v', trainable=True)
#         self.kernel_o = self.add_weight(shape=kernel_shape_v,
#                                         initializer='glorot_uniform',
#                                         name='kernel_o', trainable=True)
#         self.bias_q = self.add_weight(shape=(self.filters_q_k,),
#                                       initializer='zeros',
#                                       name='bias_q', trainable=True)
#         self.bias_k = self.add_weight(shape=(self.filters_q_k,),
#                                       initializer='zeros',
#                                       name='bias_k', trainable=True)
#         self.bias_v = self.add_weight(shape=(self.filters_v,),
#                                       initializer='zeros',
#                                       name='bias_v', trainable=True)
#         self.bias_o = self.add_weight(shape=(self.filters_v,),
#                                       initializer='zeros',
#                                       name='bias_o', trainable=True)
        super(SelfAttention, self).build(input_shape)
#         self.input_spec = InputSpec(ndim=3,
#                                     axes={2: input_shape[-1]})
#         self.built = True


    def call(self, inputs):
        if type(inputs)==type([]):
            x,masks = inputs
        else:
            x,masks = inputs,None
        q = kl.Dense(self.filters_q_k,use_bias=True)(x)
        q = kl.Activation(activation)(q)
#         q = kl.tanh(alpha=1.0)(q)
#         K.print_tensor(q, message= "q values=")
    
        k = kl.Dense(self.filters_q_k,use_bias=True)(x)
        k = kl.Activation(activation)(k)
#         k = kl.tanh(alpha=1.0)(k)
#         K.print_tensor(k, message= "k values=")
    
        v = kl.Dense(self.filters_v,use_bias=True)(x)
        v = kl.Activation(activation)(v)
#         v = kl.tanh(alpha=1.0)(v)
#         K.print_tensor(v, message= "v values=")
    
        beta = K.batch_dot(q, K.permute_dimensions(k,(0,2,1)))  # # [bs, N, N]        
#         K.print_tensor(s, message="s values=")
        

        
        if masks is not None:
#             print('apply padding')
            beta = kl.Multiply()([beta,masks])
    
#         print('s.shape:',s.shape)
        
        scores = K.softmax(beta, axis=-1)  # attention map
#         K.print_tensor(b, message= "b values=")
#         print('beta.shape:',beta.shape.as_list())
        o = K.batch_dot(scores, v)  # [bs, N, C]
#         print('o.shape:',o.shape)
#         o = K.reshape(o, shape=K.shape(x))  # [bs, h, w, C]
        
#         o = kl.tanh(alpha=1.0)(o)
#         K.print_tensor(o, message="o values=")
    
#         x = x + self.gamma * o 
#         print('x.shape:',x.shape)
        self.x_sh = tuple(x.shape.as_list())
        self.q_sh = tuple(q.shape.as_list())
        self.k_sh = tuple(k.shape.as_list())
        self.v_sh = tuple(v.shape.as_list())
        self.beta_sh = tuple(beta.shape.as_list())
        self.scores_sh = tuple(scores.shape.as_list())
        self.o_sh = tuple(o.shape.as_list())
        return [x
                ,q
                ,k
                ,v
                ,beta
                ,scores
                ,o]#, self.gamma]

    def compute_output_shape(self, input_shape):
        return [self.x_sh
                ,self.q_sh
                ,self.k_sh
                ,self.v_sh
                ,self.beta_sh
                ,self.scores_sh
                ,self.o_sh]#, tuple(self.gamma.shape.as_list())]

class CondenseAttention2D(Layer):
    '''
    input = [concatenated_multiattended_features]
    '''
    def __init__(self, ch_in, ch_out, **kwargs):
        super(CondenseAttention2D, self).__init__(**kwargs)        
        self.image_channels = ch_in 
        self.filters_o = ch_out
    def build(self, input_shape):
        kernel_shape_o = (1,1) + (self.image_channels, self.filters_o)         
        self.kernel_o = self.add_weight(shape=kernel_shape_o,
                                        initializer='glorot_uniform',
                                        name='kernel_o', trainable=True)
        self.bias_o = self.add_weight(shape=(self.filters_o,),
                                      initializer='zeros',
                                      name='bias_o', trainable=True)
        super(CondenseAttention2D, self).build(input_shape)
    def call(self, inputs):
        multihead_attended = inputs
        o = K.conv2d(multihead_attended,
                     kernel=self.kernel_o,
                     strides=(1,1), padding='same')
        o = K.bias_add(o, self.bias_o)
        o = kl.Activation(activation)(o)
        self.o_sh = tuple(o.shape.as_list())
        return o
    def compute_output_shape(self, input_shape):
        return self.o_sh    
    
    
class CondenseAttention1D(Layer):
    def __init__(self, ch_in, ch_out, **kwargs):
        super(CondenseAttention1D, self).__init__(**kwargs)        
        self.text_channels = ch_in 
        self.filters_o = ch_out
    def build(self, input_shape):
#         kernel_shape_o = (1, self.text_channels, self.filters_o)        
#         self.kernel_o = self.add_weight(shape=kernel_shape_o,
#                                         initializer='glorot_uniform',
#                                         name='kernel_o', trainable=True)
#         self.bias_o = self.add_weight(shape=(self.filters_o,),
#                                       initializer='zeros',
#                                       name='bias_o', trainable=True)
        super(CondenseAttention1D, self).build(input_shape)
    def call(self, inputs):
        multihead_attended = inputs
        o = kl.Dense(self.filters_o,use_bias=True)(multihead_attended)
        o = kl.Activation(activation)(o)
        self.o_sh = tuple(o.shape.as_list())
        return o
    def compute_output_shape(self, input_shape):
        return self.o_sh

class ResidualCombine(Layer):    
    '''
    [prev_layer, multi_attn] = inputs
    Concat attention features with previous layer features
    Use scalar gamma for weighted concatenation
    gamma1=>residual
    gamma2=>attended
    '''
    def __init__(self,method,**kwargs):
        super(ResidualCombine, self).__init__(**kwargs)
        self.method = method
    def build(self, input_shape):
#         self.gamma1 = self.add_weight(name='gamma1', shape=[1], initializer='ones', trainable=True)
#         self.gamma2 = self.add_weight(name='gamma2', shape=[1], initializer='ones', trainable=True)
        super(ResidualCombine, self).build(input_shape)        
    def call(self, inputs):
        prev_layer, multi_attn = inputs
        prev_layer_exp = K.expand_dims(axis=-1,x=prev_layer)
        multi_attn_exp = K.expand_dims(axis=-1,x=multi_attn)
        
        gamma1 = kl.Dense(1,use_bias=True)
        gamma2 = kl.Dense(1,use_bias=True)
        
        prev_layer_exp = gamma1(prev_layer_exp)
        multi_attn_exp = gamma2(multi_attn_exp)
#         g1_weight = tf.convert_to_tensor(gamma1.get_weights()[0][0])
#         g2_weight = tf.convert_to_tensor(gamma2.get_weights()[0][0])
        
        # instead of multiplying gamma weights we can do this
        prev_layer = K.squeeze(x=kl.Activation(activation)(prev_layer_exp),axis=-1)
        multi_attn = K.squeeze(x=kl.Activation(activation)(multi_attn_exp),axis=-1)
        
        if self.method == 'concat':
            x_out = kl.Concatenate(axis=-1)([prev_layer,multi_attn])    
        elif self.method == 'add':
            x_out = kl.Add()([prev_layer,multi_attn])    
        self.g1_sh = tuple(gamma1.weights[0].shape.as_list())
        self.g2_sh = tuple(gamma2.weights[0].shape.as_list())
        self.out_sh = tuple(x_out.shape.as_list())
#         print('here')
        return [x_out, gamma1.weights[0], gamma2.weights[0]]
    def compute_output_shape(self,input_shape):
        return [self.out_sh, self.g1_sh, self.g2_sh]


        
class Text2ImgCA(Layer):
    def __init__(self, img_ch, text_ch, **kwargs):
        '''
        [text, img, masks] = inputs
        text_ch: feature dimension of text
        img_ch: feature dimension of image
        output is always the shape of the input query
        query is always context aware of the key
        in this function: 
        q -> N words with d1 features
        k,v -> M image pixels with d2 features        
        '''
        super(Text2ImgCA, self).__init__(**kwargs)        
        self.text_channels = text_ch #q
        self.img_channels = img_ch #k,v
        self.filters_q = 36
        self.filters_k = 36
        self.filters_v = self.text_channels
#         self.filters_o = self.text_channels
    def build(self, input_shape):
#         kernel_shape_q = (1, self.text_channels, self.filters_q)
#         kernel_shape_k = (1, 1) + (self.img_channels, self.filters_k)
#         kernel_shape_v = (1, 1) + (self.img_channels, self.filters_v)
#         kernel_shape_o = (1, self.text_channels, self.filters_o)
#         self.N = input_shape[1]        
#         self.gamma1 = self.add_weight(name='gamma1', shape=[1], initializer='ones', trainable=True)
#         self.gamma2 = self.add_weight(name='gamma2', shape=[1], initializer='ones', trainable=True)
#         self.kernel_q = self.add_weight(shape=kernel_shape_q,
#                                         initializer='glorot_uniform',
#                                         name='kernel_q', trainable=True)
#         self.kernel_k = self.add_weight(shape=kernel_shape_k,
#                                         initializer='glorot_uniform',
#                                         name='kernel_k', trainable=True)
#         self.kernel_v = self.add_weight(shape=kernel_shape_v,
#                                         initializer='glorot_uniform',
#                                         name='kernel_v', trainable=True)
#         self.kernel_o = self.add_weight(shape=kernel_shape_o,
#                                         initializer='glorot_uniform',
#                                         name='kernel_o', trainable=True)
#         self.bias_q = self.add_weight(shape=(self.filters_q,),
#                                       initializer='zeros',
#                                       name='bias_q', trainable=True)
#         self.bias_k = self.add_weight(shape=(self.filters_k,),
#                                       initializer='zeros',
#                                       name='bias_k', trainable=True)
#         self.bias_v = self.add_weight(shape=(self.filters_v,),
#                                       initializer='zeros',
#                                       name='bias_v', trainable=True)
#         self.bias_o = self.add_weight(shape=(self.filters_o,),
#                                       initializer='zeros',
#                                       name='bias_o', trainable=True)
        super(Text2ImgCA, self).build(input_shape)
#         self.input_spec = InputSpec(ndim=3,
#                                     axes={2: input_shape[-1]})
#         self.built = True
    def call(self, inputs):
        def hw_flatten(x):
            return kl.Reshape(target_shape=(int(x.shape[1])*int(x.shape[2]),int(x.shape[3])))(x)
#             s = x.shape.as_list()
#             return K.reshape(x, shape=[-1,s[1]*s[2],s[3]])
        text, img, masks = inputs
        if masks is not None:
            self.masks = masks
#         self.text_input_shape = tuple(x1.shape[1:].as_list())
        q = kl.Dense(self.filters_q,use_bias=True)(text)
        q = kl.Activation(activation)(q)
#         q = kl.tanh(alpha=1.0)(q)
        k = kl.Conv2D(filters=self.filters_k,strides=(1,1),kernel_size=(1,1), padding='same')(img)
        k = kl.Activation(activation)(k)
#         k = kl.tanh(alpha=1.0)(k)
        v = kl.Conv2D(filters=self.filters_v,strides=(1,1),kernel_size=(1,1), padding='same')(img)
        v = kl.Activation(activation)(v)
#         v = kl.tanh(alpha=1.0)(v)
#         print('q.shape,k.shape,v.shape,',q.shape,k.shape,v.shape)
        s = K.batch_dot(q, K.permute_dimensions(hw_flatten(k), (0,2,1)))  # # [bs, N, M]
                      
        if self.masks is not None:
            beta = kl.Multiply()([s,self.masks])
        else:
            beta = s
#         print('s.shape:',s.shape)
        scores = K.softmax(beta, axis=-1)  # attention map  
                        
#         self.beta_shape = tuple(beta.shape[1:].as_list())
#         print('hw_flatten(v).shape:',hw_flatten(v).shape)
        o = K.batch_dot(scores, hw_flatten(v))  # [bs, N, C]
#         print('o.shape:',o.shape)
#         o = K.reshape(o, shape=K.shape(x2))  # [bs, h, w, C]
#         o = K.conv1d(o,
#                      kernel=self.kernel_o,
#                      strides=(1,), padding='same')
#         o = K.bias_add(o, self.bias_o)
#         o = kl.tanh(alpha=1.0)(o)
#         print('o.shape:',o.shape)
#         x_text = self.gamma1 * x1 
# #         print('x_text.shape:',x_text,x_text.shape)
#         x_att = self.gamma2 * o
# #         print('x_att.shape:',x_att,x_att.shape)
#         x_out = K.concatenate([x_text,x_att],axis=-1) #kl.Concatenate()([x_text,x_att])
#         print('x_out.shape:',x_out,x_out.shape)

        self.text_sh = tuple(text.shape.as_list())
        self.q_sh = tuple(q.shape.as_list())
        self.k_sh = tuple(k.shape.as_list())
        self.v_sh = tuple(v.shape.as_list())
        self.s_sh = tuple(s.shape.as_list())
        self.scores_sh = tuple(scores.shape.as_list())
        self.beta_sh = tuple(beta.shape.as_list())
        self.o_sh = tuple(o.shape.as_list())
        return [text,q,k,v,s,scores,beta,o]

    def compute_output_shape(self, input_shape):
#         print(input_shape)
        return [self.text_sh,self.q_sh,self.k_sh,self.v_sh,self.s_sh,self.scores_sh,self.beta_sh,self.o_sh]



class Img2TextCA(Layer):
    def __init__(self, img_ch, text_ch, **kwargs):
        '''
        text_ch: feature dimension of text
        img_ch: feature dimension of image
        output is always the shape of the input query
        query is always context aware of the key
        in this function: 
        q -> N words with d1 features
        k,v -> M image pixels with d2 features        
        '''
        super(Img2TextCA, self).__init__(**kwargs)        
        self.text_channels = text_ch #q
        self.img_channels = img_ch #k,v
        self.filters_q = 512
        self.filters_k = 512
        self.filters_v = self.img_channels
        self.filters_o = self.img_channels
    def build(self, input_shape):
        kernel_shape_q = (1,1) + (self.img_channels, self.filters_q)
        kernel_shape_k = (1, self.text_channels, self.filters_k)
        kernel_shape_v = (1, self.text_channels, self.filters_v)
        kernel_shape_o = (1,1) + (self.img_channels, self.filters_o)
#         self.N = input_shape[1]        
#         self.gamma1 = self.add_weight(name='gamma1', shape=[1], initializer='ones', trainable=True)
#         self.gamma2 = self.add_weight(name='gamma2', shape=[1], initializer='ones', trainable=True)
        self.kernel_q = self.add_weight(shape=kernel_shape_q,
                                        initializer='glorot_uniform',
                                        name='kernel_q', trainable=True)
        self.kernel_k = self.add_weight(shape=kernel_shape_k,
                                        initializer='glorot_uniform',
                                        name='kernel_k', trainable=True)
        self.kernel_v = self.add_weight(shape=kernel_shape_v,
                                        initializer='glorot_uniform',
                                        name='kernel_v', trainable=True)
        self.kernel_o = self.add_weight(shape=kernel_shape_o,
                                        initializer='glorot_uniform',
                                        name='kernel_o', trainable=True)
        self.bias_q = self.add_weight(shape=(self.filters_q,),
                                      initializer='zeros',
                                      name='bias_q', trainable=True)
        self.bias_k = self.add_weight(shape=(self.filters_k,),
                                      initializer='zeros',
                                      name='bias_k', trainable=True)
        self.bias_v = self.add_weight(shape=(self.filters_v,),
                                      initializer='zeros',
                                      name='bias_v', trainable=True)
        self.bias_o = self.add_weight(shape=(self.filters_o,),
                                      initializer='zeros',
                                      name='bias_o', trainable=True)
        super(Img2TextCA, self).build(input_shape)
#         self.input_spec = InputSpec(ndim=3,
#                                     axes={2: input_shape[-1]})
#         self.built = True
    def call(self, inputs):
        def hw_flatten(x):
            return kl.Reshape(target_shape=(int(x.shape[1])*int(x.shape[2]),int(x.shape[3])))(x)
#             s = x.shape.as_list()
#             return K.reshape(x, shape=[-1,s[1]*s[2],s[3]])
        x1, x2, masks = inputs #img, text, mask
        if masks is not None:
            self.masks = masks
#         self.text_input_shape = tuple(x1.shape[1:].as_list())
        q = K.conv2d(x1,
                     kernel=self.kernel_q,
                     strides=(1,1), padding='same')
        q = K.bias_add(q, self.bias_q)
#         q = kl.tanh(alpha=1.0)(q)
        k = K.conv1d(x2,
                     kernel=self.kernel_k,
                     strides=(1,), padding='same')
        k = K.bias_add(k, self.bias_k)
#         k = kl.tanh(alpha=1.0)(k)
        v = K.conv1d(x2,
                     kernel=self.kernel_v,
                     strides=(1,), padding='same')
        v = K.bias_add(v, self.bias_v)
#         v = kl.tanh(alpha=1.0)(v)
#         print('q.shape,k.shape,v.shape,',q.shape,k.shape,v.shape)
        s = K.batch_dot(hw_flatten(q), K.permute_dimensions(k,(0,2,1)))  # # [bs, N, M]
#         print(s.shape)
        beta = K.softmax(s, axis=-1)  # attention map
        if self.masks is not None:
            beta = K.permute_dimensions(x=beta,pattern=(0,2,1))
#             print(s.shape)
            beta = kl.Multiply()([beta,self.masks])
            beta = K.permute_dimensions(x=beta,pattern=(0,2,1))
#             print(s.shape)
#         print('s.shape:',s.shape)
        
        self.beta_shape = tuple(beta.shape[1:].as_list())
#         print('hw_flatten(v).shape:',hw_flatten(v).shape)
        o = K.batch_dot(beta, v)  # [bs, N, C]
#         print('o.shape:',o.shape)
        o = K.reshape(o, shape=K.shape(x1))  # [bs, h, w, C]
#         print('o.shape:',o.shape)
        o = K.conv2d(o,
                     kernel=self.kernel_o,
                     strides=(1,1), padding='same')
        o = K.bias_add(o, self.bias_o)
#         o = kl.tanh(alpha=1.0)(o)
#         print('o.shape:',o.shape)
#         x_text = self.gamma1 * x1 
# #         print('x_text.shape:',x_text,x_text.shape)
#         x_att = self.gamma2 * o
# #         print('x_att.shape:',x_att,x_att.shape)
#         x_out = K.concatenate([x_text,x_att],axis=-1) #kl.Concatenate()([x_text,x_att])
#         print('x_out.shape:',x_out,x_out.shape)
        self.out_sh = tuple(o.shape.as_list())
        return [o, beta]

    def compute_output_shape(self, input_shape):
#         print(input_shape)
        return [self.out_sh, self.beta_shape]#, tuple(self.gamma1.shape.as_list()), tuple(self.gamma2.shape.as_list())]


class EncDecAttention(Layer):
    def __init__(self, enc_ch, dec_ch, **kwargs):
        '''
        [dec,enc] = inputs
        dec =  q
        enc = k,v
        
        returns = [q,k,v,beta,scores,o]
        '''
        super(EncDecAttention, self).__init__(**kwargs)        
        self.enc_ch = enc_ch #q
        self.dec_ch = dec_ch #k,v
        self.filters_q = self.enc_ch//8
        self.filters_k = self.dec_ch//8
        self.filters_v = self.dec_ch
    def build(self, input_shape):
        super(EncDecAttention, self).build(input_shape)
#         self.input_spec = InputSpec(ndim=3,
#                                     axes={2: input_shape[-1]})
#         self.built = True
    def call(self, inputs):
        if len(inputs)==2:
            dec, enc = inputs
            masks = None
        elif len(inputs)==3:
            dec, enc, masks = inputs
        q = kl.Dense(self.filters_q,use_bias=True)(dec)
        q = kl.Activation(activation)(q)
#         q = kl.tanh(alpha=1.0)(q)
        k = kl.Dense(self.filters_k,use_bias=True)(enc)
        k = kl.Activation(activation)(k)
#         k = kl.tanh(alpha=1.0)(k)
        v = kl.Dense(self.filters_v,use_bias=True)(enc)
        v = kl.Activation(activation)(v)
#         v = kl.tanh(alpha=1.0)(v)
#         print('q.shape,k.shape,v.shape,',q.shape,k.shape,v.shape)
        beta = K.batch_dot(q, K.permute_dimensions(k, (0,2,1)))  # # [bs, N, M]
        
        if masks is not None:
#             print('apply padding')
            beta = kl.Multiply()([beta,masks])
        
        
        scores = K.softmax(beta, axis=-1)  # attention map  
        
        
        
        o = K.batch_dot(scores, v)  # [bs, N, C]

        self.q_sh = tuple(q.shape.as_list())
        self.k_sh = tuple(k.shape.as_list())
        self.v_sh = tuple(v.shape.as_list())
        self.beta_sh = tuple(beta.shape.as_list())
        self.scores_sh = tuple(scores.shape.as_list())
        self.o_sh = tuple(o.shape.as_list())
        return [q,k,v,beta,scores,o]

    def compute_output_shape(self, input_shape):
#         print(input_shape)
        return [self.q_sh,self.k_sh,self.v_sh,self.beta_sh,self.scores_sh,self.o_sh]


