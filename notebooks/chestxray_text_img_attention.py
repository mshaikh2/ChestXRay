#!/usr/bin/env python
# coding: utf-8

# In[3]:


import spacy
from keras.applications.vgg16 import VGG16
from keras.applications.inception_resnet_v2 import InceptionResNetV2
import matplotlib.pyplot as plt
from keras.utils import plot_model,multi_gpu_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.models import Model
from keras.models import load_model
from keras.layers import Input,GlobalAveragePooling2D,Layer,InputSpec
from keras.layers.core import Dense,Flatten,Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.utils import np_utils
import keras.backend as K
import keras.layers as kl
from keras.preprocessing import sequence
import tensorflow as tf
from tensorflow.python.client import device_lib
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from time import time
import pickle
# from AttentionModule import SelfAttention, SoftAttention
import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
import cv2
from tqdm import tqdm_notebook
from tqdm import tqdm
from AttentionMed import SelfAttention, Attention, Text2ImgCA
from time import time,localtime,strftime
# from coord import CoordinateChannel2D


# In[ ]:





# In[ ]:





# In[2]:


n_gpu=1
n_cpu=1
tf_config= tf.ConfigProto(device_count = {'GPU': n_gpu , 'CPU': n_cpu})
tf_config.gpu_options.allow_growth=True
s=tf.Session(config=tf_config)
K.set_session(s)


# In[3]:


nlp = spacy.load('en_core_web_md')


# In[5]:


path=os.getcwd() #Get the path
path


# In[6]:


proj_ds=pd.read_csv(path+'/../dataset/indiana_projections.csv')
repo_ds=pd.read_csv(path+'/../dataset/indiana_reports.csv')

display(proj_ds.head(50),proj_ds.shape)
display(repo_ds.sort_values(by='uid').head(),repo_ds.shape)


# In[7]:


f_proj_ds = proj_ds[proj_ds.projection=='Frontal']
f_proj_ds = f_proj_ds.sort_values(by='uid')
display(f_proj_ds.head())
f_proj_ds.shape


# In[8]:


c_repo_ds = repo_ds.dropna(subset=['findings','impression'],how='any')
display(c_repo_ds.head())
c_repo_ds.shape


# In[9]:


merged_ds = pd.merge(left=c_repo_ds,right=proj_ds,on='uid',how='inner')
display(merged_ds.head())
merged_ds.shape


# In[9]:


# chars = []
# char_count = {}
# finding_len = []
# max_len=0
# for finding in c_repo_ds.findings:
#     tokens = finding.lower()
#     for i in list("[-<>:.,()]/"):
#         tokens = tokens.replace(i,' ')
#     tokens=tokens.strip()
#     if max_len<len(tokens):
#         max_len=len(tokens)
#     finding_len.append(len(tokens))
#     chars+=tokens
# for impr in c_repo_ds.impression:
#     tokens = impr.lower()
#     for i in list("[-<>:.,()]/"):
#         tokens = tokens.replace(i,' ')
#     tokens=tokens.strip()
#     chars+=tokens
#     if max_len<len(tokens):
#         max_len=len(tokens)
#     finding_len.append(len(tokens))
# print(len(list(set(chars))))
# for char in chars:
#     if char not in char_count.keys():
#         char_count[char]=1
#     else:
#         char_count[char]+=1
# df_ccount = pd.DataFrame()
# df_ccount['chars']=char_count.keys()
# df_ccount['c_count']=char_count.values()
# df_ccount = df_ccount.sort_values(by='c_count',ascending=False).reset_index()
# display(df_ccount.head(10))
# df_ccount['index'].max(),max_len


# In[10]:


# ch_to_co = df_ccount
# ch_to_co.index = ch_to_co.chars
# ch_to_co = ch_to_co['index'].to_dict()
# display(ch_to_co)
# co_to_ch = df_ccount
# co_to_ch.index = co_to_ch['index']
# co_to_ch = co_to_ch['chars'].to_dict()
# display(co_to_ch)


# In[11]:


# tokens = nlp('startseq ' 
#              + ' '.join(c_repo_ds.impression.values).lower().replace('/',' ')
#              + ' endseq.' 
#              + ' startseq' 
#              + ' '.join(c_repo_ds.findings.values).lower().replace('/',' ')
#              + ' endseq.')


# In[12]:


# vocab = [str(x) for x in tokens]
# vocab


# In[13]:


# word_count = {}
# for word in vocab:
#     if word not in word_count.keys():
#         word_count[word]=1
#     else:
#         word_count[word]+=1
# df_wcount = pd.DataFrame()
# df_wcount['words']=word_count.keys()
# df_wcount['w_count']=word_count.values()
# df_wcount = df_wcount.sort_values(by='w_count',ascending=False).reset_index()
# df_wcount['index']+=1

# df_wcount.to_csv('../dataset/vocab_count.csv',index=False)
df_wcount = pd.read_csv('../dataset/vocab_count.csv')

display(df_wcount)
vocab_size=df_wcount.shape[0]
max_wlen = 193
print(vocab_size,max_wlen)


# In[14]:


new_words= [word for word in df_wcount['words'].values if not word.isalnum()]
print(len(new_words),new_words)


# In[15]:


w_to_co = df_wcount
w_to_co.index = w_to_co.words
w_to_co = w_to_co['index'].to_dict()
display(w_to_co)
co_to_w = df_wcount
co_to_w.index = co_to_w['index']
co_to_w = co_to_w['words'].to_dict()
display(co_to_w)


# In[16]:


# df = pd.DataFrame(sorted(finding_len,reverse=True))
# display(df.head(50))
# df.plot(kind='hist',figsize=(6,6))
co_to_w[13]


# In[17]:


embedding_size = 300
# embedding_matrix = np.zeros((vocab_size+1, embedding_size)) # last one : to cater to empty/-1
# for word,idx in w_to_co.items():
#     token = nlp(word)
#     embedding_matrix[idx] = token.vector
# embedding_matrix[vocab_size] = np.zeros(embedding_size)
embedding_matrix = pd.read_pickle('../dataset/initial_emb_mat.p')
embedding_matrix = embedding_matrix.values
# print(embedding_matrix.shape)
# df_emb = pd.DataFrame(embedding_matrix)
# df_emb.to_pickle('../dataset/initial_emb_mat.p')


# In[18]:


embedding_matrix.shape


# # Parameters

# In[19]:


img_arch = 'irv2'
text_arch = '1dcnn'
model_name = '{0}_{1}_text_img_attention'.format(img_arch,text_arch)
EPOCHS = 10


# # Model Initialization

# In[20]:


# def VGGNet():
#     image_input = Input(shape=(256,256,3),name='image_input')
#     # x = CoordinateChannel2D()(inp)
#     x = kl.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same', name='block1_conv1')(image_input)
#     x = kl.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same', name='block1_conv2')(x)
#     x = kl.Conv2D(filters=64, kernel_size=2, strides=2, activation='relu', padding='same', name='block1_reduction_conv')(x)
#     x = kl.BatchNormalization()(x)
#     x = kl.Dropout(0.5)(x)

#     x = kl.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same', name='block2_conv1')(x)
#     x = kl.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same', name='block2_conv2')(x)
#     x = kl.Conv2D(filters=128, kernel_size=2, strides=2, activation='relu', padding='same', name='block2_reduction_conv')(x)
#     x = kl.BatchNormalization()(x)
#     x = kl.Dropout(0.5)(x)

#     x = kl.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same', name='block3_conv1')(x)
#     x = kl.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same', name='block3_conv2')(x)
#     x = kl.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same', name='block3_conv3')(x)
#     x = kl.Conv2D(filters=256, kernel_size=2, strides=2, activation='relu', padding='same', name='block3_reduction_conv')(x)
#     x = kl.BatchNormalization()(x)
#     x = kl.Dropout(0.5)(x)

#     x = kl.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', name='block4_conv1')(x)
#     x = kl.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', name='block4_conv2')(x)
#     x = kl.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', name='block4_conv3')(x)
#     # x = CoordinateChannel2D(use_radius=True)(x)
# #     x, samap, g = SelfAttention(ch=512, name='self_attention')(x)
#     x = kl.Conv2D(filters=512, kernel_size=2, strides=2, activation='relu', padding='same', name='block4_reduction_conv')(x)
#     x = kl.BatchNormalization()(x)
#     x = kl.Dropout(0.5)(x)

#     x = kl.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', name='block5_conv1')(x)
#     x = kl.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', name='block5_conv2')(x)
#     x = kl.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', name='block5_conv3')(x)
# #     x, amaps = SoftAttention(ch=512, m=32, name='soft_attention')(x)
#     x = kl.Conv2D(filters=512, kernel_size=2, strides=2, activation='relu', padding='same', name='block5_reduction_conv')(x)
#     return Model(image_input,x,name='imgModel')
# def DenseNet():
#     qw = Input(shape=(256,256,3),name='image_input')
#     qw_1 = kl.Conv2D(strides=1,padding='valid',activation='relu',filters=64,name='conv',kernel_size=3)(qw)

#     qw_1 = densenet.densenet.conv_block(x=qw_1,growth_rate=64,name='conv_1',)

#     qw_2 = densenet.densenet.dense_block(qw_1,blocks=1,name='block_1')
#     qw_2 = kl.BatchNormalization()(qw_2)
#     qw_2 = kl.Activation('relu')(qw_2)
#     qw_2 = kl.Conv2D(filters=64, kernel_size=2, strides=2, activation='relu', padding='same', name='block1_reduction_conv')(qw_2)
#     qw_2 = kl.Dropout(0.5)(qw_2)

#     qw_2 = densenet.densenet.dense_block(qw_2,blocks=1,name='block_2')
#     qw_2 = kl.BatchNormalization()(qw_2)
#     qw_2 = kl.Activation('relu')(qw_2)
#     # qw_2 = kl.MaxPool2D(pool_size=2)(qw_2)
#     qw_2 = kl.Conv2D(filters=128, kernel_size=2, strides=2, activation='relu', padding='same', name='block2_reduction_conv')(qw_2)
#     qw_2 = kl.Dropout(0.5)(qw_2)

#     qw_2 = densenet.densenet.dense_block(qw_2,blocks=1,name='block_3')
#     qw_2 = kl.BatchNormalization()(qw_2)
#     qw_2 = kl.Activation('relu')(qw_2)
#     # qw_2 = kl.MaxPool2D(pool_size=2)(qw_2)
#     qw_2 = kl.Conv2D(filters=256, kernel_size=2, strides=2, activation='relu', padding='same', name='block3_reduction_conv')(qw_2)
#     qw_2 = kl.Dropout(0.5)(qw_2)

#     qw_2 = densenet.densenet.dense_block(qw_2,blocks=1,name='block_4')
#     qw_2 = kl.BatchNormalization()(qw_2)
#     qw_2 = kl.Activation('relu')(qw_2)
#     # qw_2 = kl.MaxPool2D(pool_size=2)(qw_2)
#     qw_2 = kl.Conv2D(filters=512, kernel_size=2, strides=2, activation='relu', padding='same', name='block4_reduction_conv')(qw_2)
#     qw_2 = kl.Dropout(0.5)(qw_2)

#     qw_2 = densenet.densenet.dense_block(qw_2,blocks=1,name='block_5')
#     qw_2 = kl.BatchNormalization()(qw_2)
#     qw_2 = kl.Activation('relu')(qw_2)
#     # qw_2 = kl.MaxPool2D(pool_size=2)(qw_2)
#     qw_2 = kl.Conv2D(filters=1024, kernel_size=2, strides=2, activation='relu', padding='same', name='block5_reduction_conv')(qw_2)
#     # qw_2 = kl.Dropout(0.5)(qw_2)
#     qw_2 = kl.BatchNormalization()(qw_2)
#     qw_2 = kl.Activation('relu')(qw_2)

#     return Model(qw,qw_2, name='imgModel')


# In[21]:


if img_arch=='vgg':
    imgNet = VGGNet()
elif img_arch == 'densenet121':
    imgNet = DenseNet121(include_top=False,input_shape=(256,256,3))
    imgNet.trainable = True
elif img_arch == 'densenet':
    imgNet = DenseNet()
elif img_arch == 'irv2':
    imgNet =InceptionResNetV2(include_top=False,input_shape=(256,256,3))
if os.path.exists('../checkpoints/irv2_1dcnn_attention_img_module.h5'.format(img_arch,text_arch)):
    print('loading existing image weights..')
    imgNet.load_weights('../checkpoints/irv2_1dcnn_attention_img_module.h5'.format(img_arch,text_arch))
imgNet.summary()


# In[22]:


def textNet():
    words_input = kl.Input(shape=(max_wlen,),name='words_input')
    padding_masks = kl.Input(shape=(max_wlen,1),name='padding_masks')
    x2 = kl.Embedding(vocab_size+1, embedding_size, mask_zero=False, name='w2v_emb')(words_input)
    xk3 = kl.Conv1D(filters=324,kernel_size=3,strides=1,activation='relu', padding='same')(x2)
    xk5 = kl.Conv1D(filters=324,kernel_size=5,strides=1,activation='relu', padding='same')(x2)
    xk7 = kl.Conv1D(filters=324,kernel_size=7,strides=1,activation='relu', padding='same')(x2)
    xk3d2 = kl.Conv1D(filters=324,kernel_size=3,strides=1,activation='relu', dilation_rate=2, padding='same')(x2)
    xk5d2 = kl.Conv1D(filters=324,kernel_size=5,strides=1,activation='relu', dilation_rate=2, padding='same')(x2)
    xk7d2 = kl.Conv1D(filters=324,kernel_size=7,strides=1,activation='relu', dilation_rate=2, padding='same')(x2)
    x2 = kl.Concatenate()([xk3,xk5,xk7,xk3d2,xk5d2,xk7d2])
    x2 = kl.BatchNormalization()(x2)
    x2 = kl.Activation('relu')(x2)
    print(x2)
#     x2 = kl.Dropout(0.1)(x2)
    sa_out_x2_1,s_x2_1,g_x2_1 = SelfAttention(ch=int(x2.shape[-1]))([x2,padding_masks])
    sa_out_x2_2,s_x2_2,g_x2_2 = SelfAttention(ch=int(x2.shape[-1]))([x2,padding_masks])
    sa_out_x2_3,s_x2_3,g_x2_3 = SelfAttention(ch=int(x2.shape[-1]))([x2,padding_masks])
    sa_out_x2_4,s_x2_4,g_x2_4 = SelfAttention(ch=int(x2.shape[-1]))([x2,padding_masks])
    print(sa_out_x2_4)
    x3 = kl.Add()([sa_out_x2_1,sa_out_x2_2,sa_out_x2_3,sa_out_x2_4])
    print(x3)
    x3 = kl.BatchNormalization()(x3)
    x3 = kl.Activation('relu')(x3)
    
    sa_out_x3_1,s_x3_1,g_x3_1 = SelfAttention(ch=int(x3.shape[-1]))([x3,padding_masks])
    sa_out_x3_2,s_x3_2,g_x3_2 = SelfAttention(ch=int(x3.shape[-1]))([x3,padding_masks])
    sa_out_x3_3,s_x3_3,g_x3_3 = SelfAttention(ch=int(x3.shape[-1]))([x3,padding_masks])
    sa_out_x3_4,s_x3_4,g_x3_4 = SelfAttention(ch=int(x3.shape[-1]))([x3,padding_masks])
    x4 = kl.Add()([sa_out_x3_1,sa_out_x3_2,sa_out_x3_3,sa_out_x3_4])
    x4 = kl.BatchNormalization()(x4)
    x4 = kl.Activation('relu')(x4)
    
    sa_out_x4_1,s_x4_1,g_x4_1 = SelfAttention(ch=int(x3.shape[-1]))([x4,padding_masks])
    sa_out_x4_2,s_x4_2,g_x4_2 = SelfAttention(ch=int(x3.shape[-1]))([x4,padding_masks])
    sa_out_x4_3,s_x4_3,g_x4_3 = SelfAttention(ch=int(x3.shape[-1]))([x4,padding_masks])
    sa_out_x4_4,s_x4_4,g_x4_4 = SelfAttention(ch=int(x3.shape[-1]))([x4,padding_masks])
    
    
    
    x5 = kl.Add()([sa_out_x4_1,sa_out_x4_2,sa_out_x4_3,sa_out_x4_4])
    x5 = kl.BatchNormalization()(x5)
    x5 = kl.Activation('relu')(x5)
#     sa_pool = kl.GlobalAveragePooling1D(name='sa_gap')(x5)
#     a_out,cs = Attention(ch=int(sa_pool.shape[-1]))([sa_pool,x2])    
#     a_pool = kl.Lambda(lambda x:K.tf.squeeze(x,axis=1),name='a_gap')(a_out)
#     out = kl.Concatenate()([a_pool,sa_pool])
#     print('cx.shape:',out.shape)
    
    return Model([words_input,padding_masks],x5,name='textModel')


# In[ ]:





# In[23]:


textNet = textNet()
textNet.summary()


# In[24]:


SVG(model_to_dot(textNet, show_layer_names=True, show_shapes=True).create(prog='dot', format='svg'))


# # Define Complete Joint Network

# In[25]:



image_input = Input(shape=(256,256,3),name='image_input')
image_features = imgNet(image_input)
image_features = kl.Dropout(0.6)(image_features)
# image_features = kl.Reshape(target_shape=(int(image_features.shape[1])*int(image_features.shape[2])
#                                           ,int(image_features.shape[3])))(image_features)
# print(image_features.shape)
# image_features = kl.Flatten()(image_features)
# image_features = kl.Dense(512,activation='relu')(image_features)

padding_masks = kl.Input(shape=(max_wlen,1),name='padding_masks')

words_input = kl.Input(shape=(max_wlen,),name='words_input')
text_features = textNet([words_input,padding_masks])
text_features = kl.Dropout(0.6)(text_features)
# text_features = kl.Dense(512,activation='relu')(text_features)
print(image_features.shape)
print(text_features.shape)


# In[26]:


# class Text2ImgCA(Layer):
#     def __init__(self, img_ch, text_ch, **kwargs):
#         '''
#         text_ch: feature dimension of text
#         img_ch: feature dimension of image
#         output is always the shape of the input query
#         query is always context aware of the key
#         in this function: 
#         q -> N words with d1 features
#         k,v -> M image pixels with d2 features        
#         '''
#         super(Text2ImgCA, self).__init__(**kwargs)        
#         self.text_channels = text_ch #q
#         self.img_channels = img_ch #k,v
#         self.filters_q = 512
#         self.filters_k = 512
#         self.filters_v = self.text_channels
#         self.filters_o = self.text_channels
#     def build(self, input_shape):
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
#         super(Text2ImgCA, self).build(input_shape)
# #         self.input_spec = InputSpec(ndim=3,
# #                                     axes={2: input_shape[-1]})
# #         self.built = True
#     def call(self, inputs):
#         def hw_flatten(x):
#             return kl.Reshape(target_shape=(int(x.shape[1])*int(x.shape[2]),int(x.shape[3])))(x)
# #             s = x.shape.as_list()
# #             return K.reshape(x, shape=[-1,s[1]*s[2],s[3]])
#         x1, x2, masks = inputs
#         if masks is not None:
#             self.padding_masks = masks
# #         self.text_input_shape = tuple(x1.shape[1:].as_list())
#         q = K.conv1d(x1,
#                      kernel=self.kernel_q,
#                      strides=(1,), padding='same')
#         q = K.bias_add(q, self.bias_q) 
#         k = K.conv2d(x2,
#                      kernel=self.kernel_k,
#                      strides=(1,1), padding='same')
#         k = K.bias_add(k, self.bias_k)
#         v = K.conv2d(x2,
#                      kernel=self.kernel_v,
#                      strides=(1,1), padding='same')
#         v = K.bias_add(v, self.bias_v)
# #         print('q.shape,k.shape,v.shape,',q.shape,k.shape,v.shape)
#         s = tf.matmul(q, hw_flatten(k), transpose_b=True)  # # [bs, N, M]
#         if self.padding_masks is not None:
#             s = tf.multiply(s,self.padding_masks)
# #         print('s.shape:',s.shape)
#         beta = K.softmax(s, axis=-1)  # attention map
#         self.beta_shape = tuple(beta.shape[1:].as_list())
# #         print('hw_flatten(v).shape:',hw_flatten(v).shape)
#         o = K.batch_dot(beta, hw_flatten(v))  # [bs, N, C]
# #         print('o.shape:',o.shape)
# #         o = K.reshape(o, shape=K.shape(x2))  # [bs, h, w, C]
#         o = K.conv1d(o,
#                      kernel=self.kernel_o,
#                      strides=(1,), padding='same')
#         o = K.bias_add(o, self.bias_o)
# #         print('o.shape:',o.shape)
#         x_text = self.gamma1 * x1 
# #         print('x_text.shape:',x_text,x_text.shape)
#         x_att = self.gamma2 * o
# #         print('x_att.shape:',x_att,x_att.shape)
#         x_out = K.concatenate([x_text,x_att],axis=-1) #kl.Concatenate()([x_text,x_att])
# #         print('x_out.shape:',x_out,x_out.shape)
#         self.out_sh = tuple(x_out.shape.as_list())
#         return [x_out, beta, self.gamma1, self.gamma2]

#     def compute_output_shape(self, input_shape):
# #         print(input_shape)
#         return [self.out_sh, self.beta_shape, tuple(self.gamma1.shape.as_list()), tuple(self.gamma2.shape.as_list())]


# In[27]:


# image_pool = kl.GlobalAveragePooling2D()(image_features)
# image_pool = kl.Dense(512,activation='relu')(image_pool)
# image_pool = kl.BatchNormalization()(image_pool)
# image_pool = kl.Activation('relu')(image_pool)
# print('image_pool.shape:',image_pool.shape)

# sa_pool = kl.GlobalAveragePooling1D(name='sa_gap')(text_features)
# print('sa_pool.shape:',sa_pool.shape)
# vantage_pool = kl.Dense(512, activation='relu')(vantage_pool)

a_out_1,beta_1,g1_1,g2_1 = Text2ImgCA(text_ch=int(text_features.shape[-1]),img_ch=int(image_features.shape[-1]))([text_features,image_features,padding_masks])
a_out_1 = kl.Activation('relu')(a_out_1)

a_out_2,beta_2,g1_2,g2_2 = Text2ImgCA(text_ch=int(text_features.shape[-1]),img_ch=int(image_features.shape[-1]))([text_features,image_features,padding_masks])
a_out_2 = kl.Activation('relu')(a_out_2)

a_out_3,beta_3,g1_3,g2_3 = Text2ImgCA(text_ch=int(text_features.shape[-1]),img_ch=int(image_features.shape[-1]))([text_features,image_features,padding_masks])
a_out_3 = kl.Activation('relu')(a_out_3)

a_out_4,beta_4,g1_4,g2_4 = Text2ImgCA(text_ch=int(text_features.shape[-1]),img_ch=int(image_features.shape[-1]))([text_features,image_features,padding_masks])
a_out_4 = kl.Activation('relu')(a_out_4)

a_conc = kl.Concatenate(name='CrossAttn_Concat')([a_out_1,a_out_2,a_out_3,a_out_4])
# print('a_conc.shape:',a_conc.shape)
a_conc_out = kl.BatchNormalization(name='concat_batchnorm')(a_conc)
a_conc_out = kl.Activation('relu')(a_conc_out)
cross_pool = kl.GlobalAveragePooling1D(name='MHCA_global_pool')(a_conc_out)
cross_pool = kl.Dense(1024, activation='relu')(cross_pool)
cross_pool = kl.BatchNormalization()(cross_pool)
cross_pool = kl.Activation('relu')(cross_pool)
# print('cross_pool.shape:',cross_pool.shape)

# x6 = kl.Conv1D(filters=512,kernel_size=3,strides=1,activation='relu',padding='same')(a_conc_out)
# x7 = kl.Conv1D(filters=512,kernel_size=5,strides=1,activation='relu',padding='same')(a_conc_out)
# x8 = kl.Conv1D(filters=512,kernel_size=7,strides=1,activation='relu',padding='same')(a_conc_out)
# x9 = kl.Conv1D(filters=512,kernel_size=3,strides=1,activation='relu',padding='same',dilation_rate=2)(a_conc_out)
# x10 = kl.Conv1D(filters=512,kernel_size=5,strides=1,activation='relu',padding='same',dilation_rate=2)(a_conc_out)
# x11 = kl.Conv1D(filters=512,kernel_size=7,strides=1,activation='relu',padding='same',dilation_rate=2)(a_conc_out)
# vantage_pool = kl.Concatenate(name='vantage_features')([x6,x7,x8,x9,x10,x11])
# vantage_pool = kl.BatchNormalization()(vantage_pool)
# vantage_pool = kl.Activation('relu')(vantage_pool)
# # Use lstm/gap for getting fixed len vector
# vantage_pool = kl.GlobalAveragePooling1D(name='vantage_pool')(vantage_pool)
# vantage_pool = kl.Dense(512, activation='relu')(vantage_pool)
# vantage_pool = kl.BatchNormalization()(vantage_pool)
# vantage_pool = kl.Activation('relu')(vantage_pool)
# print(vantage_pool.shape)

# all_concat = kl.Concatenate(name='all_concat')([sa_pool, cross_pool])#, vantage_pool])
# all_concat = kl.BatchNormalization()(all_concat)
# all_concat = kl.Activation('relu')(all_concat)
# print(all_concat.shape)


target = Dense(vocab_size+1, activation='softmax',name='target_word')(cross_pool)

model = Model([image_input,words_input,padding_masks],target)


# In[28]:


# print('New model {0}. loading word2vec embeddings'.format(model_name))
# l = textNet.get_layer('w2v_emb')
# l.set_weights([embedding_matrix])
# l.trainable = True
if not os.path.exists('../checkpoints/{0}.h5'.format(model_name)):
    print('New model {0}. loading word2vec embeddings'.format(model_name))
    l = textNet.get_layer('w2v_emb')
    l.set_weights([embedding_matrix])
    l.trainable = True
else:
    print('Existing model {0}. trained weights will be loaded'.format(model_name))
#     model.load_weights('../checkpoints/{0}.h5'.format(model_name))


# In[29]:


# w_to_co


# In[30]:


model.summary()
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001,decay=1e-6),metrics=['accuracy'])


# In[31]:


SVG(model_to_dot(model, show_layer_names=True, show_shapes=True).create(prog='dot', format='svg'))


# In[32]:


model_json = model.to_json()
with open("../model_json/{0}.json".format(model_name), "w") as json_file:
    json_file.write(model_json)


# In[33]:


emb = Model(textNet.inputs,textNet.get_layer('w2v_emb').output)
emb.summary()


# In[34]:


img = cv2.imread('../dataset/images/images_normalized/3591_IM-1770-1001-0001.dcm.png')
plt.imshow(img)


# In[35]:


def datagen(img_lst,batch_size=4):
    counter=0
    x1,x2,y,sents,masks = [],[],[],[],[]
    idx = 0
    while True:
        im = img_lst[idx]
        photo = cv2.imread('../dataset/images/images_normalized/resized_1024/{0}'.format(im))/255.0
        sent = merged_ds[merged_ds.filename==im]['findings'].values[0].lower().replace('/',' ')
        
        ts = nlp('startseq ' + sent + ' endseq')
#         print(ts)
        ts = [str(x) for x in list(ts)]
#         print(tss)
        sent_words = []
        for t in ts:
#             print(t)
#             t = 'sdadasasdasads'
            if (t not in w_to_co.keys()) : #or ('xx' in t):
#                 print('not present:',t)
                pass
            else:
                sent_words.append(t)
#         print(sent_words)
        seq = [w_to_co[x] for x in sent_words]
        # split one sequence into multiple X, y pairs
        for i in range(1, len(seq)):
            
            in_seq, out_seq = seq[:i], seq[i]
#             print(out_seq)
            in_seq = sequence.pad_sequences([in_seq], maxlen=max_wlen,padding='pre',value=0)[0]
            mask = [0+1e-8 if x<1 else  1 for x in in_seq]
            mask = np.expand_dims(mask,-1)
            out_seq = np_utils.to_categorical([out_seq], num_classes=vocab_size+1)[0]
            x1.append(photo)
            x2.append(in_seq)
            masks.append(mask)
            y.append(out_seq)
            sents.append(sent)
        counter+=1
        idx+=1
#         print(idx)
        if idx==len(img_lst):
            idx=0
        if counter==batch_size:
            counter=0
            inputs = {'image_input': np.array(x1),
                      'words_input': np.array(x2),
                      'padding_masks':np.array(masks)
                     }
            outputs = {'target_word':np.array(y),
                      'actual_sentence':np.array(sents)}
            yield inputs, outputs
            x1,x2,y,sents,masks = [],[],[],[],[]


# In[36]:




with open('../dataset/train_images.p', 'rb') as filehandle:
    # read the data as binary data stream
    train_images_list = pickle.load(filehandle)
with open('../dataset/val_images.p', 'rb') as filehandle:
    # read the data as binary data stream
    val_images_list = pickle.load(filehandle)
with open('../dataset/test_images.p', 'rb') as filehandle:
    # read the data as binary data stream
    test_images_list = pickle.load(filehandle)

print('train:',len(train_images_list),
      '+ val:',len(val_images_list),
      '+ test:',len(test_images_list),
      ' = ',(len(train_images_list)+len(val_images_list)+len(test_images_list)))
train_batch_size = 64
val_batch_size = 64
test_batch_size = 1
train_gen = datagen(train_images_list,batch_size=train_batch_size)
val_gen = datagen(val_images_list,batch_size=val_batch_size)
test_gen = datagen(test_images_list,batch_size=test_batch_size)


# In[37]:


# test_images_list = merged_ds.filename.values
# test_images_list = list(test_images_list)
# for x in train_images_list:
#     test_images_list.remove(x)
# for x in val_images_list:
#     test_images_list.remove(x)
# len(test_images_list)


# In[38]:


# imgs = merged_ds.filename.values
# np.random.shuffle(imgs)

# train_images_list = imgs[:int(0.7*len(imgs))]
# val_images_list = imgs[int(0.7*len(imgs)):int(0.8*len(imgs))]
# test_images_list = imgs[int(0.8*len(imgs)):]
# with open('../dataset/train_images.p', 'wb') as file:
#     # store the data as binary data stream
#     pickle.dump(train_images_list, file)
# with open('../dataset/val_images.p', 'wb') as file:
#     # store the data as binary data stream
#     pickle.dump(val_images_list, file)
# with open('../dataset/test_images.p', 'wb') as file:
#     # store the data as binary data stream
#     pickle.dump(test_images_list, file)
# tx,ty = next(train_gen)
# tx['padding_masks'][30].shape


# In[39]:



# display(tx['image_input'].shape
#         ,tx['words_input'][30]
#         ,tx['padding_masks'][30]
#         ,' '.join([co_to_w[x] if x !=0 else '' for x in tx['words_input'][30]])
#         ,ty['actual_sentence'][30])


# In[40]:


len(val_images_list),len(train_images_list), w_to_co['xxxx']


# In[41]:


# # def _ctc_lambda_func(args):
# #     labels, y_pred, input_length, label_length = args
# #     return K.tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)
# vggnet.summary()
# w_to_co['xxxx']
# np.clip(0,a_max=1,a_min=1e-8)


# In[42]:


print(co_to_w[273])
# print(embedding_matrix[1380])


# In[43]:


# att = np.random.randint(1,10,(4,5,5))
# mask = 1-np.triu(np.ones((4,5,5)),k=1)
# att*mask


# In[44]:


# emb_out = emb.predict(tx['words_input'])
# emb_out[1][-1]


# In[45]:


parallel_model=multi_gpu_model(model, gpus=3)

parallel_model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001,decay=1e-06),metrics=['accuracy']) #For multi GPU
parallel_model.summary()
# # model_resnet.summary()
# # parallel_model.load_weights('./../checkpoints/sa_vgg16_3d_32x24_m32_direct.h5', by_name=True, skip_mismatch=True)


# In[46]:


# mc=ModelCheckpoint(filepath='../checkpoints/vgg16_lstm_words.h5',monitor='val_loss',period=1,save_best_only=True,save_weights_only=True,mode='auto',verbose=3)
# es=EarlyStopping(patience=300,monitor='val_loss',min_delta=0.0001,mode='auto')
print('training {0} for {1} epochs'.format(model_name,EPOCHS))


# In[ ]:


from IPython.display import clear_output

# counter = 0
# model.load_weights('../checkpoints/{0}.h5'.format(model_name))
# history = model.fit_generator(train_gen
#                               ,epochs=EPOCHS
#                               ,steps_per_epoch=4
#                               ,validation_data=val_gen
#                               ,validation_steps=2
#                               ,callbacks=[mc])

hist_tl,hist_ta,hist_vl,hist_va,tt = [],[],[],[],[]
window = 16
val_window = 128
train_iterations = int(np.ceil(len(train_images_list)/train_batch_size))
val_iterations = int(np.ceil(len(val_images_list)/val_batch_size))


min_v_l = 10
t_l, t_a, v_l, v_a = 0, 0, 4.01, 0
for e in tqdm_notebook(np.arange(start=0,stop=EPOCHS),desc='Epoch'):
    start_time = time()
    tl,ta,vl,va = [],[],[],[]
    for im in tqdm_notebook(range(train_iterations),desc='Train_Iter',leave=False):
        tx,ty = next(train_gen)
        inputs1 = tx['image_input']
        inputs2 = tx['words_input']
        inputs3 = tx['padding_masks']
#         print(inputs3.shape)
        labels = ty['target_word']        
        for i in np.arange(len(inputs1),step=window):
            loss,accuracy = model.train_on_batch([inputs1[i:i+window],inputs2[i:i+window],inputs3[i:i+window]],labels[i:i+window])
            tl.append(loss)
            ta.append(accuracy)
    for im in tqdm_notebook(range(val_iterations),desc='Val_Iter',leave=False):
        vx,vy = next(val_gen)
        inputs1 = vx['image_input']
        inputs2 = vx['words_input']
        inputs3 = vx['padding_masks']
        labels = vy['target_word']
        for i in np.arange(len(inputs1),step=val_window):
            loss,accuracy = model.evaluate(verbose=0,x=[inputs1[i:i+val_window],inputs2[i:i+val_window],inputs3[i:i+window]],y=labels[i:i+val_window])
            vl.append(loss)
            va.append(accuracy)
    v_l = np.round(np.mean(vl),4)
    v_a = np.round(np.mean(va),4)
    t_l = np.round(np.mean(tl),4)
    t_a = np.round(np.mean(ta),4)
    
    hist_tl.append(t_l)
    hist_ta.append(t_a)
    hist_vl.append(v_l)
    hist_va.append(v_a)
    if v_l < min_v_l:
        min_v_l = v_l
        model.save_weights(filepath='../checkpoints/{0}.h5'.format(model_name),overwrite=True)
        imgNet.save_weights(filepath='../checkpoints/{0}_{1}_img_module.h5'.format(img_arch,text_arch),overwrite=True)
        textNet.save_weights(filepath='../checkpoints/{0}_{1}_text_module.h5'.format(img_arch,text_arch),overwrite=True)
    clear_output(wait=True)
    end_time = time()
    time_taken = end_time-start_time
    tt.append(time_taken)
    with open('../tf_runs/log.csv','a') as f:
        data = '{0:3d}/{7},{1:.4f},{2:.4f},{3:.4f},{4:.4f},{6:.4f}'.format(e+1, t_l, t_a, v_l, v_a, np.mean(tt),min_v_l,EPOCHS)
        t = strftime("%m/%d/%Y %H:%M:%S",localtime())
        f.writelines('\n[{0}],{1},{2}'.format(t, model_name,data))
    print('E:{0:3d}/{7}, tr_loss:{1:.4f}, tr_acc:{2:.4f}, v_loss:{3:.4f}, v_acc:{4:.4f} [{5:.2f} s/e] (min_v_loss:{6:.4f})'.format(e+1, t_l, t_a, v_l, v_a, np.mean(tt),min_v_l,EPOCHS))


# In[ ]:


# with open('../tf_runs/log.csv','a') as f:
#     data = '{0:3d}/{7},{1:.4f},{2:.4f},{3:.4f},{4:.4f},{6:.4f}'.format(e+1, t_l, t_a, v_l, v_a, np.mean(tt),min_v_l,EPOCHS)
#     t = strftime("%m/%d/%Y %H:%M:%S",localtime())
#     f.writelines('\n[{0}],{1},{2}'.format(t, model_name,data))
# data


# In[ ]:


history = pd.DataFrame()
history['tr_acc'] = hist_ta
history['val_acc'] = hist_va
history['tr_loss'] = hist_tl
history['val_loss'] = hist_vl
display(history[['tr_acc','val_acc']].plot())
display(history[['tr_loss','val_loss']].plot())
history.to_csv('../tf_runs/{0}.csv'.format(model_name),index=False)


# # Evaluate

# In[ ]:


model.load_weights('../checkpoints/{0}.h5'.format(model_name))


# In[ ]:



def predict_captions(image):
    start_word = ["startseq"]
    while True:
        par_caps = [w_to_co[i] for i in start_word]
        par_caps = sequence.pad_sequences([par_caps], maxlen=max_wlen, padding='pre',value=-1)
        preds = model.predict([image, np.array(par_caps)])
#         print(preds.shape)
        idx = preds.argmax(-1)
        word_pred = co_to_w[idx[0]]
#         print(par_caps)
        start_word.append(word_pred)
        
        if word_pred == "endseq" or len(start_word) > max_wlen:
            break
            
    return ' '.join(start_word[1:-1])

# print('Predicted:',' '.join(out_text))


# In[ ]:


def beam_search_predictions(image, beam_index = 3):
    start = [w_to_co["startseq"]]
    
    # start_word[0][0] = index of the starting word
    # start_word[0][1] = probability of the word predicted
    start_word = [[start, 0.0]]
    
    while len(start_word[0][0]) < max_wlen:
        temp = []
        for s in start_word:
            par_caps = sequence.pad_sequences([s[0]], maxlen=max_wlen, padding='pre', value=-1)
            preds = model.predict([image, np.array(par_caps)])
            
            # Getting the top <beam_index>(n) predictions
            word_preds = np.argsort(preds[0])[-beam_index:]
            
            # creating a new list so as to put them via the model again
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])
                    
        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word = start_word[-beam_index:]
    
    start_word = start_word[-1][0]
    intermediate_caption = [co_to_w[i] for i in start_word]

    final_caption = []
    
    for i in intermediate_caption:
        if i != 'endseq':
            final_caption.append(i)
        else:
            break
    
    final_caption = ' '.join(final_caption[1:])
    return final_caption


# In[ ]:


ref_sents = []
pred_sents = []
for counter in tqdm_notebook(range(len(test_images_list))):
    testx,testy = next(test_gen)
    photo = testx['image_input'][0]
#     plt.imshow(photo)
#     plt.show()
    photo = np.expand_dims(photo,0)
    
#     print('Actual:',testy['actual_sentence'][0])
#     print()
    # st = time()
    # pred_greedy = predict_captions(photo)
    # et = time()
    # print('Greedy Predicted:{0},[{1:.2f} s]'.format(pred_greedy,et-st))
#     st = time()
    pred_bm5 = beam_search_predictions(photo,beam_index=5)
#     et = time()
#     print('Beam-5 Predicted:{0},[{1:.2f} s]'.format(pred_bm5,et-st))
    ref_sents.append(testy['actual_sentence'][0])
    pred_sents.append(pred_bm5)
    if counter==10:
        break


# In[ ]:



# st = time()
# pred_greedy = predict_captions(photo)
# et = time()
# print('Greedy Predicted:{0},[{1:.2f} s]'.format(pred_greedy,et-st))

# st = time()
# pred_bm3 = beam_search_predictions(photo,beam_index=3)
# et = time()
# print('Beam-3 Predicted:{0},[{1:.2f} s]'.format(pred_bm3,et-st))

# st = time()
# pred_bm5 = beam_search_predictions(photo,beam_index=5)
# et = time()
# print('Beam-5 Predicted:{0},[{1:.2f} s]'.format(pred_bm5,et-st))

# st = time()
# pred_bm7 = beam_search_predictions(photo,beam_index=7)
# et = time()
# print('Beam-7 Predicted:{0},[{1:.2f} s]'.format(pred_bm7,et-st))


# In[ ]:


from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
rouge = Rouge()
reference = nlp(str(vy['actual_sentence'][0]))
reference = [[str(x) for x in list(reference)]]


# candidate = nlp(pred_greedy)
# candidate = [str(x) for x in list(candidate)]
# df_result['greedy'] = [sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
#                       ,sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
#                       ,sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
#                       ,sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))]

# candidate = nlp(pred_bm3)
# candidate = [str(x) for x in list(candidate)]
# df_result['bm3'] = [sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
#                       ,sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
#                       ,sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
#                       ,sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))]
row = []
for i in range(len(ref_sents)):
    r = ref_sents[i]
    c = pred_sents[i]
    reference = nlp(str(r))
    reference = [[str(x) for x in list(reference)]]
    candidate = nlp(str(c))
    candidate = [str(x) for x in list(candidate)]
    row.append([sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
                          ,sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
                          ,sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
                          ,sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
                          ,rouge.get_scores(hyps=c,refs=r)[0]['rouge-l']['f']])


# candidate = nlp(pred_bm7)
# candidate = [str(x) for x in list(candidate)]
# df_result['bm7'] = [sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
#                       ,sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
#                       ,sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
#                       ,sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))]
df_result = pd.DataFrame(row)
df_result.columns = ['BLEU-1','BLEU-2','BLEU-3','BLEU-4','ROUGE-L']
df_result.round(3)


# In[ ]:


ref_sents,pred_sents


# In[ ]:


pd.DataFrame(row)


# In[ ]:



reference = str(vy['actual_sentence'][0])
# reference = [[str(x) for x in list(reference)]]
candidate = pred_greedy
# candidate = [str(x) for x in list(candidate)]
ro = rouge.get_scores(hyps=candidate,refs=reference)
ro[0]['rouge-l']['f']


# In[ ]:


pd.DataFrame(ro[0]).T


# In[ ]:


from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor


# In[ ]:


def calc_scores(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
#         (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
#         (Meteor(),"METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores 


# In[ ]:


calc_scores(ref=reference,hypo=candidate)


# In[45]:


all_text = merged_ds['findings'].values


# In[46]:


lengths = []
for text in all_text:
    words = text.split(' ')
    lengths.append(len(words))


# In[47]:


df_lengths = pd.DataFrame(lengths,columns=['lengths'])


# In[69]:


df_lengths[df_lengths['lengths']>80].shape


# In[59]:


df_lengths.mean(),df_lengths.median(),df_lengths.mode()


# In[71]:


55/6450


# In[ ]:




