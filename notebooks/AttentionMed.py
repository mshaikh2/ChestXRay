import keras.backend as K
import keras.layers as kl
from keras.layers import Layer,InputSpec,Conv1D 
from keras import regularizers, constraints, initializers, activations
from keras.layers.recurrent import Recurrent
import tensorflow as tf

    
class Attention(Layer):
    def __init__(self,timesteps, ch, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.channels = ch
        self.timesteps = timesteps
        self.filters_q_k = self.channels       
        self.filters_v = self.channels
        

    def build(self, input_shape):
#         print('build input_shape:',input_shape)
        kernel_shape_q_k = (self.timesteps, self.channels, self.filters_q_k)
#         print('kernel_shape_q_k:',kernel_shape_q_k)
        kernel_shape_v = (self.timesteps, self.channels, self.filters_v)
#         print('kernel_shape_v:',kernel_shape_v)
        self.N = input_shape[1]  
#         print('self.N:',self.N)
#         self.gamma = self.add_weight(name='att_gamma', shape=[1], initializer='zeros', trainable=True)
        self.kernel_q = self.add_weight(shape=kernel_shape_q_k,
                                        initializer='glorot_uniform',
                                        name='kernel_q', trainable=True)
        self.kernel_k = self.add_weight(shape=kernel_shape_q_k,
                                        initializer='glorot_uniform',
                                        name='kernel_k', trainable=True)
        self.kernel_v = self.add_weight(shape=kernel_shape_v,
                                        initializer='glorot_uniform',
                                        name='kernel_v', trainable=True)
        self.kernel_o = self.add_weight(shape=kernel_shape_v,
                                        initializer='glorot_uniform',
                                        name='kernel_o', trainable=True)
        self.bias_q = self.add_weight(shape=(self.filters_q_k,),
                                      initializer='zeros',
                                      name='bias_q', trainable=True)
        self.bias_k = self.add_weight(shape=(self.filters_q_k,),
                                      initializer='zeros',
                                      name='bias_k', trainable=True)
        self.bias_v = self.add_weight(shape=(self.filters_v,),
                                      initializer='zeros',
                                      name='bias_v', trainable=True)
        self.bias_o = self.add_weight(shape=(self.filters_v,),
                                      initializer='zeros',
                                      name='bias_o', trainable=True)
        super(Attention, self).build(input_shape)
#         self.input_spec = InputSpec(shape=[(None,None,None),(None,None,None)])
#         self.built = True
        
    def call(self, inputs):
        h,x = inputs # context_vector, self-attended out
#         print('before:',h.shape)
        if len(h.shape)<3:
            h = K.expand_dims(axis=1,x=h)
#         print('after:',h.shape)
        q = K.conv1d(h,
                     kernel=self.kernel_q,
                     strides=(1,), padding='same')
        q = K.bias_add(q, self.bias_q)
#         q = kl.BatchNormalization()(q)
        q = kl.Activation('relu')(q)
        k = K.conv1d(x,
                     kernel=self.kernel_k,
                     strides=(1,), padding='same')
        k = K.bias_add(k, self.bias_k)
#         k = kl.BatchNormalization()(k)
        k = kl.Activation('relu')(k)
        v = K.conv1d(x,
                     kernel=self.kernel_v,
                     strides=(1,), padding='same')
        v = K.bias_add(v, self.bias_v)
#         v = kl.BatchNormalization()(v)
        v = kl.Activation('relu')(v)
#         v = kl.ELU(alpha=1.0)(v)
#         print('q.shape,k.shape,v.shape,',q.shape,k.shape,v.shape)
        s = K.batch_dot(q, K.permute_dimensions(k,(0,2,1)))  # # [bs, 1, N]
    
#         print('s.shape:',s.shape)
        beta = K.softmax(s, axis=-1)  # attention map (bs, 1, N)
        self.beta_shape = tuple(beta.shape[1:].as_list())
#         print('beta.shape:',beta.shape)
        
        o = K.batch_dot(beta, v)  # [bs, 1, C]
        
#         print('o.shape:',o.shape)
        o = K.conv1d(o,
                     kernel=self.kernel_o,
                     strides=(1,), padding='same')
        o = K.bias_add(o, self.bias_o)
#         o = kl.BatchNormalization()(o)
        o = kl.Activation('relu')(o)
#         o = kl.ELU(alpha=1.0)(o)
#         print('o.shape:',o.shape)
#         o = K.reshape(o, shape=K.shape(x))  # [bs, h, w, C]
#         h = h + self.gamma * o
        self.out_shape = tuple(o.shape.as_list())
#         print(self.o_shape)
#         print('x.shape:',x.shape)
        return [o, s]#, self.gamma]

    def compute_output_shape(self, input_shape):
#         print(self.o_shape,self.beta_shape, tuple(self.gamma.shape.as_list()))
        return [self.out_shape, self.beta_shape]#, tuple(self.gamma.shape.as_list())] 

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
        q = kl.Dense(self.filters_q_k)(x)
        q = kl.Activation('relu')(q)
#         q = kl.ELU(alpha=1.0)(q)
#         K.print_tensor(q, message= "q values=")
    
        k = kl.Dense(self.filters_q_k)(x)
        k = kl.Activation('relu')(k)
#         k = kl.ELU(alpha=1.0)(k)
#         K.print_tensor(k, message= "k values=")
    
        v = kl.Dense(self.filters_v)(x)
        v = kl.Activation('relu')(v)
#         v = kl.ELU(alpha=1.0)(v)
#         K.print_tensor(v, message= "v values=")
    
        s = K.batch_dot(q, K.permute_dimensions(k,(0,2,1)))  # # [bs, N, N]        
#         K.print_tensor(s, message="s values=")
        

        
        if masks is not None:
            print('apply padding')
            beta = kl.Multiply()([s,masks])
        else:
            beta = s
#         print('s.shape:',s.shape)
        
        scores = K.softmax(beta, axis=-1)  # attention map
#         K.print_tensor(b, message= "b values=")
#         print('beta.shape:',beta.shape.as_list())
        o = K.batch_dot(scores, v)  # [bs, N, C]
#         print('o.shape:',o.shape)
#         o = K.reshape(o, shape=K.shape(x))  # [bs, h, w, C]
        
#         o = kl.ELU(alpha=1.0)(o)
#         K.print_tensor(o, message="o values=")
    
#         x = x + self.gamma * o 
#         print('x.shape:',x.shape)
        self.x_sh = tuple(x.shape.as_list())
        self.q_sh = tuple(q.shape.as_list())
        self.k_sh = tuple(k.shape.as_list())
        self.v_sh = tuple(v.shape.as_list())
        self.s_sh = tuple(s.shape.as_list())
        self.scores_sh = tuple(scores.shape.as_list())
        self.beta_sh = tuple(beta.shape.as_list())
        self.o_sh = tuple(o.shape.as_list())
        return [x,q,k,v,s,scores,beta,o]#, self.gamma]

    def compute_output_shape(self, input_shape):
        return [self.x_sh,self.q_sh,self.k_sh,self.v_sh,self.s_sh,self.scores_sh,self.beta_sh,self.o_sh]#, tuple(self.gamma.shape.as_list())]

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
        o = kl.Activation('relu')(o)
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
        o = kl.Dense(self.filters_o)(multihead_attended)
        o = kl.Activation('relu')(o)
        self.o_sh = tuple(o.shape.as_list())
        return o
    def compute_output_shape(self, input_shape):
        return self.o_sh

class ResidualCombine(Layer):    
    '''
    [previous_layer_features, attended_features]
    Concat attention features with previous layer features
    Use scalar gamma for weighted concatenation
    gamma1=>residual
    gamma2=>attended
    '''
    def __init__(self,method,**kwargs):
        super(ResidualCombine, self).__init__(**kwargs)
        self.method = method
    def build(self, input_shape):
        self.gamma1 = self.add_weight(name='gamma1', shape=[1], initializer='ones', trainable=True)
        self.gamma2 = self.add_weight(name='gamma2', shape=[1], initializer='ones', trainable=True)
        super(ResidualCombine, self).build(input_shape)        
    def call(self, inputs):
        prev_layer, multi_attn = inputs
        g1 = kl.Activation('relu')(self.gamma1)
        g2 = kl.Activation('relu')(self.gamma2)
        prev_layer = g1 * prev_layer 
        multi_attn = g2 * multi_attn   
        if self.method == 'concat':
            x_out = kl.Concatenate(axis=-1)([prev_layer,multi_attn])    
        elif self.method == 'add':
            x_out = kl.Add()([prev_layer,multi_attn])    
        self.out_sh = tuple(x_out.shape.as_list())
        return [x_out, self.gamma1, self.gamma2]
    def compute_output_shape(self,input_shape):
        return [self.out_sh, tuple(self.gamma1.shape.as_list()), tuple(self.gamma2.shape.as_list())]

        
class Text2ImgCA(Layer):
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
        super(Text2ImgCA, self).__init__(**kwargs)        
        self.text_channels = text_ch #q
        self.img_channels = img_ch #k,v
        self.filters_q = 512
        self.filters_k = 512
        self.filters_v = self.text_channels
        self.filters_o = self.text_channels
    def build(self, input_shape):
        kernel_shape_q = (1, self.text_channels, self.filters_q)
        kernel_shape_k = (1, 1) + (self.img_channels, self.filters_k)
        kernel_shape_v = (1, 1) + (self.img_channels, self.filters_v)
        kernel_shape_o = (1, self.text_channels, self.filters_o)
        self.N = input_shape[1]        
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
        super(Text2ImgCA, self).build(input_shape)
#         self.input_spec = InputSpec(ndim=3,
#                                     axes={2: input_shape[-1]})
#         self.built = True
    def call(self, inputs):
        def hw_flatten(x):
            return kl.Reshape(target_shape=(int(x.shape[1])*int(x.shape[2]),int(x.shape[3])))(x)
#             s = x.shape.as_list()
#             return K.reshape(x, shape=[-1,s[1]*s[2],s[3]])
        x1, x2, masks = inputs
        if masks is not None:
            self.padding_masks = masks
#         self.text_input_shape = tuple(x1.shape[1:].as_list())
        q = K.conv1d(x1,
                     kernel=self.kernel_q,
                     strides=(1,), padding='same')
        q = K.bias_add(q, self.bias_q) 
#         q = kl.ELU(alpha=1.0)(q)
        k = K.conv2d(x2,
                     kernel=self.kernel_k,
                     strides=(1,1), padding='same')
        k = K.bias_add(k, self.bias_k)
#         k = kl.ELU(alpha=1.0)(k)
        v = K.conv2d(x2,
                     kernel=self.kernel_v,
                     strides=(1,1), padding='same')
        v = K.bias_add(v, self.bias_v)
#         v = kl.ELU(alpha=1.0)(v)
#         print('q.shape,k.shape,v.shape,',q.shape,k.shape,v.shape)
        s = K.batch_dot(q, K.permute_dimensions(hw_flatten(k), (0,2,1)))  # # [bs, N, M]
        beta = K.softmax(s, axis=-1)  # attention map                
        if self.padding_masks is not None:
            beta = kl.Multiply()([beta,self.padding_masks])
#         print('s.shape:',s.shape)
        
                        
        self.beta_shape = tuple(beta.shape[1:].as_list())
#         print('hw_flatten(v).shape:',hw_flatten(v).shape)
        o = K.batch_dot(beta, hw_flatten(v))  # [bs, N, C]
#         print('o.shape:',o.shape)
#         o = K.reshape(o, shape=K.shape(x2))  # [bs, h, w, C]
        o = K.conv1d(o,
                     kernel=self.kernel_o,
                     strides=(1,), padding='same')
        o = K.bias_add(o, self.bias_o)
        o = kl.ELU(alpha=1.0)(o)
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
            self.padding_masks = masks
#         self.text_input_shape = tuple(x1.shape[1:].as_list())
        q = K.conv2d(x1,
                     kernel=self.kernel_q,
                     strides=(1,1), padding='same')
        q = K.bias_add(q, self.bias_q)
#         q = kl.ELU(alpha=1.0)(q)
        k = K.conv1d(x2,
                     kernel=self.kernel_k,
                     strides=(1,), padding='same')
        k = K.bias_add(k, self.bias_k)
#         k = kl.ELU(alpha=1.0)(k)
        v = K.conv1d(x2,
                     kernel=self.kernel_v,
                     strides=(1,), padding='same')
        v = K.bias_add(v, self.bias_v)
#         v = kl.ELU(alpha=1.0)(v)
#         print('q.shape,k.shape,v.shape,',q.shape,k.shape,v.shape)
        s = K.batch_dot(hw_flatten(q), K.permute_dimensions(k,(0,2,1)))  # # [bs, N, M]
#         print(s.shape)
        beta = K.softmax(s, axis=-1)  # attention map
        if self.padding_masks is not None:
            beta = K.permute_dimensions(x=beta,pattern=(0,2,1))
#             print(s.shape)
            beta = kl.Multiply()([beta,self.padding_masks])
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
#         o = kl.ELU(alpha=1.0)(o)
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


