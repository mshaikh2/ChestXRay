
# coding: utf-8

# In[ ]:


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
from keras.optimizers import Adam,SGD
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
os.environ["CUDA_VISIBLE_DEVICES"]='0,1,3'
import cv2
from tqdm import tqdm_notebook
from tqdm import tqdm
from AttentionMed import SelfAttention, Attention, Text2ImgCA, Img2TextCA, ResidualCombine1D, ResidualCombine2D
from time import time,localtime,strftime
# from coord import CoordinateChannel2D


# In[2]:


n_gpu=3
n_cpu=1
tf_config= tf.ConfigProto(device_count = {'GPU': n_gpu , 'CPU': n_cpu})
tf_config.gpu_options.allow_growth=True
s=tf.Session(config=tf_config)
K.set_session(s)


# In[3]:


nlp = spacy.load('en_core_web_md')


# In[4]:


path=os.getcwd() #Get the path
path


# In[5]:


proj_ds=pd.read_csv(path+'/../dataset/indiana_projections.csv')
repo_ds=pd.read_csv(path+'/../dataset/indiana_reports.csv')

# display(proj_ds.head(50),proj_ds.shape)
# display(repo_ds.sort_values(by='uid').head(),repo_ds.shape)


# In[6]:


f_proj_ds = proj_ds[proj_ds.projection=='Frontal']
f_proj_ds = f_proj_ds.sort_values(by='uid')
# display(f_proj_ds.head())
f_proj_ds.shape


# In[7]:


c_repo_ds = repo_ds.dropna(subset=['findings','impression'],how='any')
# display(c_repo_ds.head())
c_repo_ds.shape


# In[8]:


merged_ds = pd.merge(left=c_repo_ds,right=proj_ds,on='uid',how='inner')
# display(merged_ds.head())
merged_ds.shape


df_wcount = pd.read_csv('../dataset/vocab_count.csv')

# display(df_wcount)
vocab_size=df_wcount.shape[0]
max_wlen = 60
print(vocab_size,max_wlen)


# In[18]:


new_words= [word for word in df_wcount['words'].values if not word.isalnum()]
print(len(new_words),new_words)


# In[19]:


w_to_co = df_wcount 
w_to_co.index = w_to_co.words
w_to_co = w_to_co['index'].to_dict()
# display(w_to_co)
co_to_w = df_wcount
co_to_w.index = co_to_w['index']
co_to_w = co_to_w['words'].to_dict()
# display(co_to_w)
co_to_w[13]


# In[21]:


embedding_size = 300
embedding_matrix = pd.read_pickle('../dataset/initial_emb_mat.p')
embedding_matrix = embedding_matrix.values

img_arch = 'irv2'
text_arch = '1dcnn'
model_name = '{0}_{1}_text_img_attention'.format(img_arch,text_arch)
EPOCHS = 50
lr = 0.0001
dropout_rate=0.8
elu_alpha = 1.0


if img_arch=='vgg':
    imgNet = VGGNet()
elif img_arch == 'densenet121':
    imgNet = DenseNet121(include_top=False,input_shape=(256,256,3))
    imgNet.trainable = True
elif img_arch == 'densenet':
    imgNet = DenseNet()
elif img_arch == 'irv2':
    imgNet =InceptionResNetV2(include_top=False,input_shape=(256,256,3))
    imgNet.trainable = True
if os.path.exists('../checkpoints/irv2_1dcnn_attention_img_module.h5'.format(img_arch,text_arch)):
    print('loading existing image weights..')
    imgNet.load_weights('../checkpoints/irv2_1dcnn_attention_img_module.h5'.format(img_arch,text_arch))
imgNet.summary()

def getTextNet():
    words_input = kl.Input(shape=(max_wlen,),name='words_input')
    padding_masks = kl.Input(shape=(max_wlen,1),name='padding_masks')
    x2 = kl.Embedding(vocab_size+1, embedding_size, mask_zero=False, name='w2v_emb')(words_input)
    xk3 = kl.Conv1D(filters=324,kernel_size=3,strides=1,padding='same')(x2)
    xk3 = kl.ELU(alpha=elu_alpha)(xk3)
    xk5 = kl.Conv1D(filters=324,kernel_size=5,strides=1,padding='same')(x2)
    xk5 = kl.ELU(alpha=elu_alpha)(xk5)
    xk7 = kl.Conv1D(filters=324,kernel_size=7,strides=1,padding='same')(x2)
    xk7 = kl.ELU(alpha=elu_alpha)(xk7)
    xk3d2 = kl.Conv1D(filters=324,kernel_size=3,strides=1,dilation_rate=2, padding='same')(x2)
    xk3d2 = kl.ELU(alpha=elu_alpha)(xk3d2)
    xk5d2 = kl.Conv1D(filters=324,kernel_size=5,strides=1,dilation_rate=2, padding='same')(x2)
    xk5d2 = kl.ELU(alpha=elu_alpha)(xk5d2)
    xk7d2 = kl.Conv1D(filters=324,kernel_size=7,strides=1,dilation_rate=2, padding='same')(x2)
    xk7d2 = kl.ELU(alpha=elu_alpha)(xk7d2)
    x2 = kl.Concatenate()([xk3,xk5,xk7,xk3d2,xk5d2,xk7d2])
#     x2 = kl.BatchNormalization()(x2)
#     x2 = kl.ELU(alpha=elu_alpha)(x2)
    x2 = kl.Conv1D(filters=100,kernel_size=1,strides=1,padding='same')(x2)
    x2 = kl.BatchNormalization()(x2)
    x2 = kl.ELU(alpha=elu_alpha)(x2)
    # print('x2.shape:',x2.shape)
    x2 = kl.Dropout(dropout_rate)(x2)
    sa_out_x2_1,s_x2_1 = SelfAttention(ch=int(x2.shape[-1]),name='sa_2_1')([x2,padding_masks])
    sa_out_x2_2,s_x2_2 = SelfAttention(ch=int(x2.shape[-1]),name='sa_2_2')([x2,padding_masks])
    sa_out_x2_3,s_x2_3 = SelfAttention(ch=int(x2.shape[-1]),name='sa_2_3')([x2,padding_masks])
    sa_out_x2_4,s_x2_4 = SelfAttention(ch=int(x2.shape[-1]),name='sa_2_4')([x2,padding_masks])
#     print(sa_out_x2_4)
    x3 = kl.Concatenate(name='concat_sa_2')([sa_out_x2_1,sa_out_x2_2,sa_out_x2_3,sa_out_x2_4])
    # x3 = kl.ELU(alpha=elu_alpha,name='act_concat_sa_2')(x3)
    x3comb,x3_g1,x3_g2 = ResidualCombine1D(ch_in=int(x3.shape[-1]),ch_out=100)([x2,x3])
    x3comb = kl.BatchNormalization()(x3comb)
    # x3comb = kl.ELU(alpha=elu_alpha)(x3comb)
    x3comb = kl.Conv1D(filters=100,kernel_size=1,strides=1,padding='same')(x3comb)
    x3comb = kl.BatchNormalization()(x3comb)
    x3comb = kl.ELU()(x3comb)
    
    
    
    
    x3comb = kl.Dropout(dropout_rate)(x3comb)
    
    sa_out_x3_1,s_x3_1 = SelfAttention(ch=int(x3comb.shape[-1]),name='sa_3_1')([x3comb,padding_masks])
    sa_out_x3_2,s_x3_2 = SelfAttention(ch=int(x3comb.shape[-1]),name='sa_3_2')([x3comb,padding_masks])
    sa_out_x3_3,s_x3_3 = SelfAttention(ch=int(x3comb.shape[-1]),name='sa_3_3')([x3comb,padding_masks])
    sa_out_x3_4,s_x3_4 = SelfAttention(ch=int(x3comb.shape[-1]),name='sa_3_4')([x3comb,padding_masks])
    x4 = kl.Concatenate(name='concat_sa_3')([sa_out_x3_1,sa_out_x3_2,sa_out_x3_3,sa_out_x3_4])
    # x4 = kl.ELU(alpha=elu_alpha,name='act_concat_sa_3')(x4)
    x4comb,x4_g1,x4_g2 = ResidualCombine1D(ch_in=int(x4.shape[-1]),ch_out=100)([x3comb,x4])
    x4comb = kl.BatchNormalization()(x4comb)
    # x4comb = kl.ELU(alpha=elu_alpha)(x4comb)
    x4comb = kl.Conv1D(filters=100,kernel_size=1,strides=1,padding='same')(x4comb)
    x4comb = kl.BatchNormalization()(x4comb)
    x4comb = kl.ELU(alpha=elu_alpha)(x4comb)
    # x4comb = kl.Dropout(dropout_rate)(x4comb)
    
    # sa_out_x4_1,s_x4_1 = SelfAttention(ch=int(x4comb.shape[-1]),name='sa_4_1')([x4comb,padding_masks])
    # sa_out_x4_2,s_x4_2 = SelfAttention(ch=int(x4comb.shape[-1]),name='sa_4_2')([x4comb,padding_masks])
    # sa_out_x4_3,s_x4_3 = SelfAttention(ch=int(x4comb.shape[-1]),name='sa_4_3')([x4comb,padding_masks])
    # sa_out_x4_4,s_x4_4 = SelfAttention(ch=int(x4comb.shape[-1]),name='sa_4_4')([x4comb,padding_masks])
    # x5 = kl.Concatenate(name='concat_sa_4')([sa_out_x4_1,sa_out_x4_2,sa_out_x4_3,sa_out_x4_4])
    # x5 = kl.ELU(alpha=elu_alpha,name='act_concat_sa_4')(x5)
    # x5comb,x5_g1,x5_g2 = ResidualCombine1D(ch_in=int(x5.shape[-1]),ch_out=256)([x4comb,x5]) 
    # x5comb = kl.BatchNormalization()(x5comb)
    # x5comb = kl.ELU(alpha=elu_alpha)(x5comb)
    # x5comb = kl.Conv1D(filters=256,kernel_size=1,strides=1,padding='same')(x5comb)
    # x5comb = kl.BatchNormalization()(x5comb)
    # x5comb = kl.ELU(alpha=elu_alpha)(x5comb)
    
    return Model([words_input,padding_masks],x4comb,name='textModel')


# In[141]:


textNet = None
textNet = getTextNet()
textNet.summary()


# In[143]:


# SVG(model_to_dot(textNet, show_layer_names=True, show_shapes=True).create(prog='dot', format='svg'))


# # Define Complete Joint Network

# In[144]:



image_input = Input(shape=(256,256,3),name='image_input')
image_features = imgNet(image_input)
image_features = kl.Dropout(dropout_rate)(image_features)

padding_masks = kl.Input(shape=(max_wlen,1),name='padding_masks')
words_input = kl.Input(shape=(max_wlen,),name='words_input')

text_features = textNet([words_input,padding_masks])
text_features = kl.Dropout(dropout_rate)(text_features)
print(image_features.shape)
print(text_features.shape)


# In[145]:


a_out_im_1,beta_im_1 = Img2TextCA(text_ch=int(text_features.shape[-1]),img_ch=int(image_features.shape[-1]))([image_features,text_features,padding_masks])
# a_out_im_1 = kl.ELU(alpha=elu_alpha)(a_out_im_1)
a_out_im_2,beta_im_2 = Img2TextCA(text_ch=int(text_features.shape[-1]),img_ch=int(image_features.shape[-1]))([image_features,text_features,padding_masks])
# a_out_im_2 = kl.ELU(alpha=elu_alpha)(a_out_im_2)
a_out_im_3,beta_im_3 = Img2TextCA(text_ch=int(text_features.shape[-1]),img_ch=int(image_features.shape[-1]))([image_features,text_features,padding_masks])
# a_out_im_3 = kl.ELU(alpha=elu_alpha)(a_out_im_3)
a_out_im_4,beta_im_4 = Img2TextCA(text_ch=int(text_features.shape[-1]),img_ch=int(image_features.shape[-1]))([image_features,text_features,padding_masks])
# a_out_im_4 = kl.ELU(alpha=elu_alpha)(a_out_im_4)
a_conc_im_out = kl.Concatenate(name='img2text_concat')([a_out_im_1,a_out_im_2,a_out_im_3,a_out_im_4])
# print('a_conc_im_out.shape:',a_conc_im_out.shape)

img2text_comb,g1,g2 = ResidualCombine2D(ch_in=int(a_conc_im_out.shape[-1]),ch_out=512)([image_features,a_conc_im_out])
img2text_comb = kl.BatchNormalization(name='img2text_comb_batchnorm')(img2text_comb)
img2text_comb = kl.ELU(alpha=elu_alpha)(img2text_comb)
img2text_comb = kl.Dropout(dropout_rate)(img2text_comb)
img2text_pool = kl.GlobalAveragePooling2D(name='img2text_global_pool')(img2text_comb)


# In[146]:


a_out_1,beta_1 = Text2ImgCA(text_ch=int(text_features.shape[-1]),img_ch=int(image_features.shape[-1]))([text_features,image_features,padding_masks])
# a_out_1 = kl.ELU(alpha=elu_alpha)(a_out_1)
# print(a_out_1.shape)
a_out_2,beta_2 = Text2ImgCA(text_ch=int(text_features.shape[-1]),img_ch=int(image_features.shape[-1]))([text_features,image_features,padding_masks])
# a_out_2 = kl.ELU(alpha=elu_alpha)(a_out_2)
# print(a_out_2.shape)
a_out_3,beta_3 = Text2ImgCA(text_ch=int(text_features.shape[-1]),img_ch=int(image_features.shape[-1]))([text_features,image_features,padding_masks])
# a_out_3 = kl.ELU(alpha=elu_alpha)(a_out_3)
# print(a_out_3.shape)
a_out_4,beta_4 = Text2ImgCA(text_ch=int(text_features.shape[-1]),img_ch=int(image_features.shape[-1]))([text_features,image_features,padding_masks])
# a_out_4 = kl.ELU(alpha=elu_alpha)(a_out_4)
# print(a_out_4.shape)
a_conc_out = kl.Concatenate(name='text2img_concat')([a_out_1,a_out_2,a_out_3,a_out_4])
# print('a_conc_out.shape:',a_conc_out.shape)

text2img_comb,g1,g2 = ResidualCombine1D(ch_in=int(a_conc_out.shape[-1]),ch_out=512)([text_features,a_conc_out])
text2img_comb = kl.BatchNormalization(name='text2img_comb_batchnorm')(text2img_comb)
text2img_comb = kl.ELU(alpha=elu_alpha)(text2img_comb)
text2img_comb = kl.Dropout(dropout_rate)(text2img_comb)
text2img_pool = kl.GlobalAveragePooling1D(name='text2img_global_pool')(text2img_comb)


# In[147]:


all_concat = kl.Concatenate()([img2text_pool,text2img_pool])
cross_pool = kl.Dense(512)(all_concat)
cross_pool = kl.ELU(alpha=elu_alpha)(cross_pool)
cross_pool = kl.Dropout(dropout_rate)(cross_pool)
target = Dense(vocab_size+1, activation='softmax',name='target_word')(cross_pool)
model = Model([image_input,words_input,padding_masks],target)


# In[148]:



if not os.path.exists('../checkpoints/{0}.h5'.format(model_name)):
    print('New model {0}. loading word2vec embeddings'.format(model_name))
    l = textNet.get_layer('w2v_emb')
    l.set_weights([embedding_matrix])
    l.trainable = False
else:
    print('Existing model {0}. trained weights will be loaded'.format(model_name))
    model.load_weights('../checkpoints/{0}.h5'.format(model_name))
    # textNet.load_weights('../checkpoints/irv2_1dcnn_attention_text_module.h5')
    # imgNet.load_weights('../checkpoints/irv2_1dcnn_attention_img_module.h5')

# In[106]:


# w_to_co



# In[149]:


model.summary()
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr),metrics=['accuracy'])
# model.load_weights('../checkpoints/{0}.h5'.format(model_name))


# In[108]:


# SVG(model_to_dot(model, show_layer_names=True, show_shapes=True).create(prog='dot', format='svg'))


# In[109]:


model_json = model.to_json()
with open("../model_json/{0}.json".format(model_name), "w") as json_file:
    json_file.write(model_json)


# In[110]:


emb = Model(textNet.inputs[0],textNet.get_layer('w2v_emb').output)
emb.summary()


# In[111]:


# img = cv2.imread('../dataset/images/images_normalized/3591_IM-1770-2561-0001.dcm.png')
# plt.imshow(img)


# In[40]:


def datagen(img_lst,batch_size=4):
    counter=0
    x1,x2,y,sents,masks, img_names = [],[],[],[],[],[]
    idx = 0
    while True:
        im = img_lst[idx]
        
        photo = cv2.imread('../dataset/images/images_normalized/resized_1024/{0}'.format(im))/255.0
        sent = merged_ds[merged_ds.filename==im]['findings'].values[0].lower().replace('/',' ')
        
        ts = nlp(sent)
#         print(ts)
        ts = [str(x) for x in list(ts)]
#         print(tss)
        ts = ts[:max_wlen-1]
        sent_words = ['startseq']
        for t in ts:
#             print(t)
#             t = 'sdadasasdasads'
            if (t not in w_to_co.keys()) : #or ('xx' in t):
#                 print('not present:',t)
                pass
            else:
                sent_words.append(t)
        sent_words.append('endseq')
#         print(sent_words)

        seq = [w_to_co[x] for x in sent_words]
        # split one sequence into multiple X, y pairs
        for i in range(1, len(seq)):
            
            in_seq, out_seq = seq[:i], seq[i]
#             print(out_seq)
            in_seq = sequence.pad_sequences([in_seq], maxlen=max_wlen,padding='pre',value=0)[0]
            mask = [0.000001 if x<1 else  1 for x in in_seq]
            mask = np.expand_dims(mask,-1)
            out_seq = np_utils.to_categorical([out_seq], num_classes=vocab_size+1)[0]
            x1.append(photo)
            img_names.append(im)
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
                      'padding_masks':np.array(masks),
                      'image_names':np.array(img_names)
                     }
            outputs = {'target_word':np.array(y),
                      'actual_sentence':np.array(sents)}
            yield inputs, outputs
            x1,x2,y,sents,masks,img_names = [],[],[],[],[],[]


# In[112]:




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


# In[42]:


# test_images_list = merged_ds.filename.values
# test_images_list = list(test_images_list)
# for x in train_images_list:
#     test_images_list.remove(x)
# for x in val_images_list:
#     test_images_list.remove(x)
# len(test_images_list)


# In[43]:


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


# In[44]:



# display(tx['image_input'].shape
#         ,tx['words_input'][30]
#         ,tx['padding_masks'][30]
#         ,' '.join([co_to_w[x] if x !=0 else '' for x in tx['words_input'][30]])
#         ,ty['actual_sentence'][30])


# In[45]:


print(len(val_images_list),len(train_images_list), w_to_co['xxxx'])


# In[46]:


# # def _ctc_lambda_func(args):
# #     labels, y_pred, input_length, label_length = args
# #     return K.tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)
# vggnet.summary()
# w_to_co['xxxx']
# np.clip(0,a_max=1,a_min=1e-8)


# In[47]:


# print(co_to_w[273])
# print(embedding_matrix[1380])
# for u,v in zip(tx['words_input'][0],tx['padding_masks'][0]):
#     print(u,v)


# In[48]:


# att = np.random.randint(1,10,(4,5,5))
# mask = 1-np.triu(np.ones((4,5,5)),k=1)
# att*mask
# tx,ty = next(train_gen)


# In[49]:


# for i in range(200):
    # print('---',i,'---')
    # print('['+tx['image_names'][i]+'], input_seq:',', '.join(['<None> {0}'.format(y) if x==0 else '<{0}> {1}'.format(co_to_w[x],y) for x,y in zip(tx['words_input'][i],tx['padding_masks'][i])]),', \t next_word:',co_to_w[np.argmax(ty['target_word'][i])])


# In[50]:


# print(tx['words_input'][1])
# embedding_matrix[105]


# In[51]:


# emb_out = emb.predict(tx['words_input'])
# emb_out[1][-2]


# In[ ]:


parallel_model=multi_gpu_model(model, gpus=3)
# # model_resnet.summary()
# # parallel_model.load_weights('./../checkpoints/sa_vgg16_3d_32x24_m32_direct.h5', by_name=True, skip_mismatch=True)


# In[ ]:


parallel_model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=lr),metrics=['accuracy']) #For multi GPU
parallel_model.summary()


# In[ ]:


# mc=ModelCheckpoint(filepath='../checkpoints/vgg16_lstm_words.h5',monitor='val_loss',period=1,save_best_only=True,save_weights_only=True,mode='auto',verbose=3)
# es=EarlyStopping(patience=300,monitor='val_loss',min_delta=0.0001,mode='auto')

print('training {0} for {1} epochs'.format(model_name,EPOCHS))
# lr = K.eval(parallel_model.optimizer.lr)
# decay = 1e-3


# In[ ]:


def CheckSelfAttention(input_data,labels):
    words_input,padding_masks=input_data
    for i in range(words_input.shape[0]):
        print('-*-*-',i,'-*-*-')
        print('input_seq:',', '.join(['<None> {0}'.format(y) if x==0 else '<{0}> {1}'.format(co_to_w[x],y) for x,y in zip(words_input[i],padding_masks[i])]),', \t next_word:',co_to_w[np.argmax(labels[i])])
    model_concat_sa_2 = Model(textNet.inputs,textNet.get_layer('act_concat_sa_2').output)
    model_concat_sa_3 = Model(textNet.inputs,textNet.get_layer('act_concat_sa_3').output)
    model_concat_sa_4 = Model(textNet.inputs,textNet.get_layer('act_concat_sa_4').output)
    
    return [model_concat_sa_2.predict(input_data)
            , model_concat_sa_3.predict(input_data)
            , model_concat_sa_4.predict(input_data)
            , textNet.predict(input_data)]
#     print(concat_sa_2)
    
def ReduceLROnPlateau(decay=0.5):
    old_lr = K.eval(parallel_model.optimizer.lr)
    new_lr = old_lr * decay
    return old_lr,new_lr


# In[ ]:


# ReduceLROnPlateau()


# In[ ]:


patience = 2
patience_counter = 0
min_delta = 0.001


# In[129]:


# from IPython.display import clear_output

# counter = 0

# history = model.fit_generator(train_gen
#                               ,epochs=EPOCHS
#                               ,steps_per_epoch=4
#                               ,validation_data=val_gen
#                               ,validation_steps=2
#                               ,callbacks=[mc])

hist_tl,hist_ta,hist_vl,hist_va,tt,lr_arr = [],[],[],[],[],[]
window = 48
val_window = 128
train_iterations = int(np.ceil(len(train_images_list)/train_batch_size))
val_iterations = int(np.ceil(len(val_images_list)/val_batch_size))

print('train_iterations:{0}; val_iterations:{1}'.format(train_iterations,val_iterations))
min_v_l = 20
t_l, t_a, v_l, v_a = 0, 0, 4.01, 0
train_gen = datagen(train_images_list,batch_size=train_batch_size)
for e in tqdm(np.arange(start=0,stop=EPOCHS),desc='Epoch'):
    start_time = time()
    tl,ta,vl,va = [],[],[],[]
    for im in tqdm(range(train_iterations),desc='Train_Iter',leave=False):
        train_gen = datagen(train_images_list,batch_size=train_batch_size)
        tx,ty = next(train_gen)
        inputs1 = tx['image_input']
        inputs2 = tx['words_input']
        inputs3 = tx['padding_masks']
#         print(inputs3.shape)
        labels = ty['target_word']        
        for i in np.arange(len(inputs1),step=window):
            # print('train step 1')
            # loss,accuracy = parallel_model.train_on_batch([inputs1[i:i+window],inputs2[i:i+window],inputs3[i:i+window]],labels[i:i+window])
            # print('train step 2')
            # loss,accuracy = parallel_model.train_on_batch([inputs1[i:i+window],inputs2[i:i+window],inputs3[i:i+window]],labels[i:i+window])
            # print('train step 3')
            # loss,accuracy = parallel_model.train_on_batch([inputs1[i:i+window],inputs2[i:i+window],inputs3[i:i+window]],labels[i:i+window])
            # print('train step 4')
            # loss,accuracy = parallel_model.train_on_batch([inputs1[i:i+window],inputs2[i:i+window],inputs3[i:i+window]],labels[i:i+window])
            # print('train step 5')
            # loss,accuracy = parallel_model.train_on_batch([inputs1[i:i+window],inputs2[i:i+window],inputs3[i:i+window]],labels[i:i+window])
            # print('train step 6')
            # loss,accuracy = parallel_model.train_on_batch([inputs1[i:i+window],inputs2[i:i+window],inputs3[i:i+window]],labels[i:i+window])
            # print('train step 7')
            # loss,accuracy = parallel_model.train_on_batch([inputs1[i:i+window],inputs2[i:i+window],inputs3[i:i+window]],labels[i:i+window])
            # print('train step 8')
            # loss,accuracy = parallel_model.train_on_batch([inputs1[i:i+window],inputs2[i:i+window],inputs3[i:i+window]],labels[i:i+window])
            # print('train step 9')
            loss,accuracy = parallel_model.train_on_batch([inputs1[i:i+window],inputs2[i:i+window],inputs3[i:i+window]],labels[i:i+window])
            # sa_2,sa_3,sa_4,textNet_output = CheckSelfAttention([inputs2[i:i+window],inputs3[i:i+window]],labels[i:i+window])            
            
            tl.append(loss)
            ta.append(accuracy)
            # break
        # break
    # break
    for im in tqdm(range(val_iterations),desc='Val_Iter',leave=False):
        vx,vy = next(val_gen)
        inputs1 = vx['image_input']
        inputs2 = vx['words_input']
        inputs3 = vx['padding_masks']
        labels = vy['target_word']
        for i in np.arange(len(inputs1),step=val_window):
            loss,accuracy = parallel_model.evaluate(verbose=0,x=[inputs1[i:i+val_window],inputs2[i:i+val_window],inputs3[i:i+val_window]],y=labels[i:i+val_window])
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
    if len(hist_vl)>=3:
        if hist_vl[-2]-hist_vl[-1]<min_delta:
            patience_counter+=1
            if patience==patience_counter:
                old_lr,lr = ReduceLROnPlateau()
                patience_counter = 0
                K.set_value(parallel_model.optimizer.lr,lr)
                K.set_value(model.optimizer.lr,lr)
    lr_arr.append(lr)
    if v_l < min_v_l:
        min_v_l = v_l
        model.save_weights(filepath='../checkpoints/{0}.h5'.format(model_name),overwrite=True)
        imgNet.save_weights(filepath='../checkpoints/{0}_{1}_img_module.h5'.format(img_arch,text_arch),overwrite=True)
        textNet.save_weights(filepath='../checkpoints/{0}_{1}_text_module.h5'.format(img_arch,text_arch),overwrite=True)
#    clear_output(wait=True)
    end_time = time()
    time_taken = end_time-start_time
    tt.append(time_taken)
    with open('../tf_runs/log.csv','a') as f:
        data = '{0:3d}/{7},{1:.4f},{2:.4f},{3:.4f},{4:.4f},{6:.4f},{8:.6f}'.format(e+1, t_l, t_a, v_l, v_a, np.mean(tt),min_v_l,EPOCHS,lr)
        t = strftime("%m/%d/%Y %H:%M:%S",localtime())
        f.writelines('\n[{0}],{1},{2}'.format(t, model_name,data))
    print('E:{0:3d}/{7}, tr_loss:{1:.4f}, tr_acc:{2:.4f}, v_loss:{3:.4f}, v_acc:{4:.4f}, lr:{8:.6f} [{5:.2f} s/e] (min_v_loss:{6:.4f})'.format(e+1, t_l, t_a, v_l, v_a, np.mean(tt),min_v_l,EPOCHS,lr))


# In[130]:


# print(pd.DataFrame(sa_2[1]))


# # In[131]:


# print(pd.DataFrame(sa_3[1]))


# # In[132]:


# print(pd.DataFrame(sa_4[1]))


# # In[133]:


# print(pd.DataFrame(textNet_output[1]))


# In[77]:


# with open('../tf_runs/log.csv','a') as f:
#     data = '{0:3d}/{7},{1:.4f},{2:.4f},{3:.4f},{4:.4f},{6:.4f}'.format(e+1, t_l, t_a, v_l, v_a, np.mean(tt),min_v_l,EPOCHS)
#     t = strftime("%m/%d/%Y %H:%M:%S",localtime())
#     f.writelines('\n[{0}],{1},{2}'.format(t, model_name,data))
# data


# In[ ]:


# history = pd.DataFrame()
# history['tr_acc'] = hist_ta
# history['val_acc'] = hist_va
# history['tr_loss'] = hist_tl
# history['val_loss'] = hist_vl
# display(history[['tr_acc','val_acc']].plot())
# display(history[['tr_loss','val_loss']].plot())
# history.to_csv('../tf_runs/{0}_x_delete.csv'.format(model_name),index=False)


# # Evaluate

# In[ ]:


# model.load_weights('../checkpoints/{0}.h5'.format(model_name))


# # In[ ]:



# def predict_captions(image):
    # start_word = ["startseq"]
    # mask = np.zeros((1,max_wlen,1))+1e-8
    # counter=0
    # while True:
        # counter-=1
        # mask[:,counter,:]=1.0
        # print(np.sum(mask))
        # par_caps = [w_to_co[i] for i in start_word]
        # par_caps = sequence.pad_sequences([par_caps], maxlen=max_wlen, padding='pre',value=0)
        # preds = model.predict([image, np.array(par_caps),mask])
# #         print(preds.shape)
        # idx = preds.argmax(-1)
        # word_pred = co_to_w[idx[0]]
# #         print(par_caps)
        # start_word.append(word_pred)
        
        # if word_pred == "endseq" or len(start_word) > max_wlen:
            # break
            
    # return ' '.join(start_word[1:-1])

# # print('Predicted:',' '.join(out_text))


# # In[ ]:


# def beam_search_predictions(image, beam_index = 3):
    # start = [w_to_co["startseq"]]
    
    # # start_word[0][0] = index of the starting word
    # # start_word[0][1] = probability of the word predicted
    # start_word = [[start, 0.0]]
    # mask = np.zeros((1,max_wlen,1))+1e-8
    # counter=0
    # while len(start_word[0][0]) < max_wlen:
        # temp = []
        # counter-=1
        # mask[:,counter,:]=1.0
        # print(np.sum(mask))
        # for s in start_word:
            
            # par_caps = sequence.pad_sequences([s[0]], maxlen=max_wlen, padding='pre', value=0)
            # preds = model.predict([image, np.array(par_caps),mask])
            
            # # Getting the top <beam_index>(n) predictions
            # word_preds = np.argsort(preds[0])[-beam_index:]
            
            # # creating a new list so as to put them via the model again
            # for w in word_preds:
                # next_cap, prob = s[0][:], s[1]
                # next_cap.append(w)
                # prob += preds[0][w]
                # temp.append([next_cap, prob])
                    
        # start_word = temp
        # # Sorting according to the probabilities
        # start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # # Getting the top words
        # start_word = start_word[-beam_index:]
    
    # start_word = start_word[-1][0]
    # intermediate_caption = [co_to_w[i] for i in start_word]

    # final_caption = []
    
    # for i in intermediate_caption:
        # if i != 'endseq':
            # final_caption.append(i)
        # else:
            # break
    
    # final_caption = ' '.join(final_caption[1:])
    # return final_caption


# # In[ ]:


# ref_sents = []
# pred_sents = []
# for counter in tqdm_notebook(range(len(test_images_list))):
    # testx,testy = next(test_gen)
    # photo = testx['image_input'][0]
# #     plt.imshow(photo)
# #     plt.show()
    # photo = np.expand_dims(photo,0)
    
# #     print('Actual:',testy['actual_sentence'][0])
# #     print()
    # # st = time()
    # # pred_greedy = predict_captions(photo)
    # # et = time()
    # # print('Greedy Predicted:{0},[{1:.2f} s]'.format(pred_greedy,et-st))
# #     st = time()
    # pred_bm5 = predict_captions(photo)
# #     et = time()
# #     print('Beam-5 Predicted:{0},[{1:.2f} s]'.format(pred_bm5,et-st))
    # ref_sents.append(testy['actual_sentence'][0])
    # pred_sents.append(pred_bm5)
    # if counter==10:
        # break


# # In[ ]:



# # st = time()
# # pred_greedy = predict_captions(photo)
# # et = time()
# # print('Greedy Predicted:{0},[{1:.2f} s]'.format(pred_greedy,et-st))

# # st = time()
# # pred_bm3 = beam_search_predictions(photo,beam_index=3)
# # et = time()
# # print('Beam-3 Predicted:{0},[{1:.2f} s]'.format(pred_bm3,et-st))

# # st = time()
# # pred_bm5 = beam_search_predictions(photo,beam_index=5)
# # et = time()
# # print('Beam-5 Predicted:{0},[{1:.2f} s]'.format(pred_bm5,et-st))

# # st = time()
# # pred_bm7 = beam_search_predictions(photo,beam_index=7)
# # et = time()
# # print('Beam-7 Predicted:{0},[{1:.2f} s]'.format(pred_bm7,et-st))

# ref_sents,pred_sents


# # In[ ]:


# from nltk.translate.bleu_score import sentence_bleu
# from rouge import Rouge
# rouge = Rouge()
# reference = nlp(str(vy['actual_sentence'][0]))
# reference = [[str(x) for x in list(reference)]]


# # candidate = nlp(pred_greedy)
# # candidate = [str(x) for x in list(candidate)]
# # df_result['greedy'] = [sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
# #                       ,sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
# #                       ,sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
# #                       ,sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))]

# # candidate = nlp(pred_bm3)
# # candidate = [str(x) for x in list(candidate)]
# # df_result['bm3'] = [sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
# #                       ,sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
# #                       ,sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
# #                       ,sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))]
# row = []
# for i in range(len(ref_sents)):
    # r = ref_sents[i]
    # c = pred_sents[i]
    # reference = nlp(str(r))
    # reference = [[str(x) for x in list(reference)]]
    # candidate = nlp(str(c))
    # candidate = [str(x) for x in list(candidate)]
    # row.append([sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
                          # ,sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
                          # ,sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
                          # ,sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
                          # ,rouge.get_scores(hyps=c,refs=r)[0]['rouge-l']['f']])


# # candidate = nlp(pred_bm7)
# # candidate = [str(x) for x in list(candidate)]
# # df_result['bm7'] = [sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
# #                       ,sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
# #                       ,sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
# #                       ,sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))]
# df_result = pd.DataFrame(row)
# df_result.columns = ['BLEU-1','BLEU-2','BLEU-3','BLEU-4','ROUGE-L']
# df_result.round(3)


# # In[ ]:


# ref_sents,pred_sents


# # In[ ]:


# pd.DataFrame(row)


# # In[ ]:



# reference = str(vy['actual_sentence'][0])
# # reference = [[str(x) for x in list(reference)]]
# candidate = pred_greedy
# # candidate = [str(x) for x in list(candidate)]
# ro = rouge.get_scores(hyps=candidate,refs=reference)
# ro[0]['rouge-l']['f']


# # In[ ]:


# pd.DataFrame(ro[0]).T


# # In[ ]:


# from pycocoevalcap.bleu.bleu import Bleu
# from pycocoevalcap.rouge.rouge import Rouge
# from pycocoevalcap.cider.cider import Cider
# from pycocoevalcap.meteor.meteor import Meteor


# # In[ ]:


# def calc_scores(ref, hypo):
    # """
    # ref, dictionary of reference sentences (id, sentence)
    # hypo, dictionary of hypothesis sentences (id, sentence)
    # score, dictionary of scores
    # """
    # scorers = [
# #         (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
# #         (Meteor(),"METEOR"),
        # (Rouge(), "ROUGE_L"),
        # (Cider(), "CIDEr")
    # ]
    # final_scores = {}
    # for scorer, method in scorers:
        # score, scores = scorer.compute_score(ref, hypo)
        # if type(score) == list:
            # for m, s in zip(method, score):
                # final_scores[m] = s
        # else:
            # final_scores[method] = score
    # return final_scores 


# # In[ ]:


# calc_scores(ref=reference,hypo=candidate)


# # In[ ]:


# from nlgeval import NLGEval
# nlgeval = NLGEval()  # loads the models


# # In[ ]:


# references = []
# hypothesis = []
# import pickle
# import numpy as np
# with open('../dataset/chestxray_cnn_attention_decoder_ref_sents.p', 'rb') as file:
    # # store the data as binary data stream
    # references=pickle.load(file)
# with open('../dataset/chestxray_cnn_attention_decoder_pred_sents.p', 'rb') as file:
    # # store the data as binary data stream
    # hypothesis=pickle.load(file)
    
# references=np.array(references)
# hypothesis=np.array(hypothesis)
# references[:2],hypothesis[:2]


# # In[ ]:


# refs=np.expand_dims(references,1)
# # hyps=np.expand_dims(hypothesis,0)
# refs=refs.tolist()
# # hyps=hyps.tolist()
# len(refs),len(hypothesis)


# # In[ ]:


# metrics_dict = nlgeval.compute_metrics(refs, hypothesis)


# # In[ ]:


# r = {idx: strippedlines for (idx, strippedlines) in enumerate(refs)}


# # In[ ]:


# len(r)


# # In[ ]:


# h = {idx: [lines.strip()] for (idx, lines) in enumerate(hypothesis)}


# # In[ ]:


# len(h)


# # In[ ]:


# refs[0],hypothesis[0]


# # In[ ]:


# from nlgeval import compute_individual_metrics


# # In[ ]:


# metrics_dict = compute_individual_metrics(refs[1], hypothesis[1])


# # In[ ]:


# import pandas as pd
# df = pd.DataFrame.from_dict(metrics_dict,orient='index').T


# # In[ ]:


# df = pd.DataFrame()
# for i in range(300):
    # m = nlgeval.compute_individual_metrics(refs[i], hypothesis[i])
    # d = pd.DataFrame.from_dict(m,orient='index').T
    # df = df.append(d)
    # print(np.around(i*256//300,2),end='\r')


# # In[ ]:


# df.shape


# # In[ ]:


# df.mean()


# # In[ ]:


# 0.98/2

