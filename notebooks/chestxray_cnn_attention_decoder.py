#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
from AttentionMed import SelfAttention, Attention
# from coord import CoordinateChannel2D


# In[6]:





# In[ ]:





# In[2]:


n_gpu=1
n_cpu=1
tf_config= tf.ConfigProto(device_count = {'GPU': n_gpu , 'CPU': n_cpu})
tf_config.gpu_options.allow_growth=True
s=tf.Session(config=tf_config)
K.set_session(s)


# In[ ]:


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


# In[10]:


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


# In[11]:


# ch_to_co = df_ccount
# ch_to_co.index = ch_to_co.chars
# ch_to_co = ch_to_co['index'].to_dict()
# display(ch_to_co)
# co_to_ch = df_ccount
# co_to_ch.index = co_to_ch['index']
# co_to_ch = co_to_ch['chars'].to_dict()
# display(co_to_ch)


# In[12]:


# tokens = nlp('startseq ' 
#              + ' '.join(c_repo_ds.impression.values).lower().replace('/',' ')
#              + ' endseq.' 
#              + ' startseq' 
#              + ' '.join(c_repo_ds.findings.values).lower().replace('/',' ')
#              + ' endseq.')


# In[13]:


# vocab = [str(x) for x in tokens]
# vocab


# In[14]:


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


# In[15]:


new_words= [word for word in df_wcount['words'].values if not word.isalnum()]
len(new_words)


# In[16]:


w_to_co = df_wcount
w_to_co.index = w_to_co.words
w_to_co = w_to_co['index'].to_dict()
display(w_to_co)
co_to_w = df_wcount
co_to_w.index = co_to_w['index']
co_to_w = co_to_w['words'].to_dict()
display(co_to_w)


# In[17]:


# df = pd.DataFrame(sorted(finding_len,reverse=True))
# display(df.head(50))
# df.plot(kind='hist',figsize=(6,6))
co_to_w[13]


# In[18]:


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


# In[19]:


embedding_matrix.shape


# # Parameters

# In[292]:


img_arch = 'irv2'
text_arch = '1dcnn_attention'
model_name = '{0}_{1}_words'.format(img_arch,text_arch)
EPOCHS = 10


# # Model Initialization

# In[297]:


def VGGNet():
    image_input = Input(shape=(256,256,3),name='image_input')
    # x = CoordinateChannel2D()(inp)
    x = kl.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same', name='block1_conv1')(image_input)
    x = kl.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same', name='block1_conv2')(x)
    x = kl.Conv2D(filters=64, kernel_size=2, strides=2, activation='relu', padding='same', name='block1_reduction_conv')(x)
    x = kl.BatchNormalization()(x)
    x = kl.Dropout(0.5)(x)

    x = kl.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same', name='block2_conv1')(x)
    x = kl.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same', name='block2_conv2')(x)
    x = kl.Conv2D(filters=128, kernel_size=2, strides=2, activation='relu', padding='same', name='block2_reduction_conv')(x)
    x = kl.BatchNormalization()(x)
    x = kl.Dropout(0.5)(x)

    x = kl.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same', name='block3_conv1')(x)
    x = kl.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same', name='block3_conv2')(x)
    x = kl.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same', name='block3_conv3')(x)
    x = kl.Conv2D(filters=256, kernel_size=2, strides=2, activation='relu', padding='same', name='block3_reduction_conv')(x)
    x = kl.BatchNormalization()(x)
    x = kl.Dropout(0.5)(x)

    x = kl.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', name='block4_conv1')(x)
    x = kl.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', name='block4_conv2')(x)
    x = kl.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', name='block4_conv3')(x)
    # x = CoordinateChannel2D(use_radius=True)(x)
#     x, samap, g = SelfAttention(ch=512, name='self_attention')(x)
    x = kl.Conv2D(filters=512, kernel_size=2, strides=2, activation='relu', padding='same', name='block4_reduction_conv')(x)
    x = kl.BatchNormalization()(x)
    x = kl.Dropout(0.5)(x)

    x = kl.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', name='block5_conv1')(x)
    x = kl.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', name='block5_conv2')(x)
    x = kl.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', name='block5_conv3')(x)
#     x, amaps = SoftAttention(ch=512, m=32, name='soft_attention')(x)
    x = kl.Conv2D(filters=512, kernel_size=2, strides=2, activation='relu', padding='same', name='block5_reduction_conv')(x)
    return Model(image_input,x,name='imgModel')
def DenseNet():
    qw = Input(shape=(256,256,3),name='image_input')
    qw_1 = kl.Conv2D(strides=1,padding='valid',activation='relu',filters=64,name='conv',kernel_size=3)(qw)

    qw_1 = densenet.densenet.conv_block(x=qw_1,growth_rate=64,name='conv_1',)

    qw_2 = densenet.densenet.dense_block(qw_1,blocks=1,name='block_1')
    qw_2 = kl.BatchNormalization()(qw_2)
    qw_2 = kl.Activation('relu')(qw_2)
    qw_2 = kl.Conv2D(filters=64, kernel_size=2, strides=2, activation='relu', padding='same', name='block1_reduction_conv')(qw_2)
    qw_2 = kl.Dropout(0.5)(qw_2)

    qw_2 = densenet.densenet.dense_block(qw_2,blocks=1,name='block_2')
    qw_2 = kl.BatchNormalization()(qw_2)
    qw_2 = kl.Activation('relu')(qw_2)
    # qw_2 = kl.MaxPool2D(pool_size=2)(qw_2)
    qw_2 = kl.Conv2D(filters=128, kernel_size=2, strides=2, activation='relu', padding='same', name='block2_reduction_conv')(qw_2)
    qw_2 = kl.Dropout(0.5)(qw_2)

    qw_2 = densenet.densenet.dense_block(qw_2,blocks=1,name='block_3')
    qw_2 = kl.BatchNormalization()(qw_2)
    qw_2 = kl.Activation('relu')(qw_2)
    # qw_2 = kl.MaxPool2D(pool_size=2)(qw_2)
    qw_2 = kl.Conv2D(filters=256, kernel_size=2, strides=2, activation='relu', padding='same', name='block3_reduction_conv')(qw_2)
    qw_2 = kl.Dropout(0.5)(qw_2)

    qw_2 = densenet.densenet.dense_block(qw_2,blocks=1,name='block_4')
    qw_2 = kl.BatchNormalization()(qw_2)
    qw_2 = kl.Activation('relu')(qw_2)
    # qw_2 = kl.MaxPool2D(pool_size=2)(qw_2)
    qw_2 = kl.Conv2D(filters=512, kernel_size=2, strides=2, activation='relu', padding='same', name='block4_reduction_conv')(qw_2)
    qw_2 = kl.Dropout(0.5)(qw_2)

    qw_2 = densenet.densenet.dense_block(qw_2,blocks=1,name='block_5')
    qw_2 = kl.BatchNormalization()(qw_2)
    qw_2 = kl.Activation('relu')(qw_2)
    # qw_2 = kl.MaxPool2D(pool_size=2)(qw_2)
    qw_2 = kl.Conv2D(filters=1024, kernel_size=2, strides=2, activation='relu', padding='same', name='block5_reduction_conv')(qw_2)
    # qw_2 = kl.Dropout(0.5)(qw_2)
    qw_2 = kl.BatchNormalization()(qw_2)
    qw_2 = kl.Activation('relu')(qw_2)

    return Model(qw,qw_2, name='imgModel')


# In[2]:


if img_arch=='vgg':
    imgNet = VGGNet()
elif img_arch == 'densenet121':
    imgNet = DenseNet121(include_top=False,input_shape=(256,256,3))
    imgNet.trainable = True
elif img_arch == 'densenet':
    imgNet = DenseNet()
elif img_arch == 'irv2':
    imgNet =InceptionResNetV2(include_top=False,input_shape=(256,256,3))
if os.path.exists('../checkpoints/{0}_{1}_img_module.h5'.format(img_arch,text_arch)):
    imgNet.load_weights('../checkpoints/{0}_{1}_img_module.h5'.format(img_arch,text_arch))
imgNet.summary()


# In[299]:


def textNet():
    words_input = kl.Input(shape=(max_wlen,),name='words_input')
    x2 = kl.Embedding(vocab_size+1, embedding_size, mask_zero=False, name='w2v_emb')(words_input)
    x2 = kl.Conv1D(filters=512,kernel_size=3,strides=1,activation='relu', padding='valid')(x2)
    x2 = kl.Dropout(0.1)(x2)
    sa_out,s,g = SelfAttention(ch=int(x2.shape[-1]))(x2)
    sa_pool = kl.GlobalAveragePooling1D(name='sa_gap')(sa_out)
    sa_pool = kl.Dense(512,activation='relu')(sa_pool)
#     print(x3.shape,x.shape)
    a_out,cs = Attention(ch=int(sa_pool.shape[-1]))([sa_pool,sa_out])    
    a_pool = kl.Lambda(lambda x:K.tf.squeeze(x,axis=1),name='a_gap')(a_out)
    out = kl.Concatenate()([a_pool,sa_pool])
#     print('cx.shape:',out.shape)
    
    return Model(words_input,out,name='textModel')


# In[300]:


textNet = textNet()
textNet.summary()


# In[301]:


SVG(model_to_dot(textNet, show_layer_names=True, show_shapes=True).create(prog='dot', format='svg'))


# # Define Complete Joint Network

# In[302]:



image_input = Input(shape=(256,256,3),name='image_input')
image_features = imgNet(image_input)
image_features = kl.Dropout(0.1)(image_features)
image_features = kl.Flatten()(image_features)
image_features = kl.Dense(512,activation='relu')(image_features)

words_input = kl.Input(shape=(max_wlen,),name='words_input')
text_features = textNet(words_input)
text_features = kl.Dropout(0.1)(text_features)
text_features = kl.Dense(512,activation='relu')(text_features)

g = kl.Concatenate()([image_features,text_features])
g = kl.Dense(512, activation='relu')(g)
target = Dense(vocab_size+1, activation='softmax',name='target_word')(g)
model = Model([image_input,words_input],target)


# In[303]:


print('New model {0}. loading word2vec embeddings'.format(model_name))
l = textNet.get_layer('w2v_emb')
l.set_weights([embedding_matrix])
l.trainable = True
if not os.path.exists('../checkpoints/{0}.h5'.format(model_name)):
    print('New model {0}. loading word2vec embeddings'.format(model_name))
    l = textNet.get_layer('w2v_emb')
    l.set_weights([embedding_matrix])
    l.trainable = True
else:
    print('Existing model {0}. trained weights will be loaded'.format(model_name))
#     model.load_weights('../checkpoints/{0}.h5'.format(model_name))


# In[304]:


# w_to_co


# In[330]:


model.summary()
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001,decay=1e-6),metrics=['accuracy'])


# In[307]:


SVG(model_to_dot(model, show_layer_names=True, show_shapes=True).create(prog='dot', format='svg'))


# In[308]:


model_json = model.to_json()
with open("../model_json/{0}.json".format(model_name), "w") as json_file:
    json_file.write(model_json)


# In[309]:


emb = Model(textNet.inputs,textNet.get_layer('w2v_emb').output)
emb.summary()


# In[310]:


img = cv2.imread('../dataset/images/images_normalized/3591_IM-1770-1001-0001.dcm.png')
plt.imshow(img)


# In[311]:


def datagen(img_lst,batch_size=4):
    counter=0
    x1,x2,y,sents = [],[],[],[]
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
            # split into input and output pair
            in_seq, out_seq = seq[:i], seq[i]
#             print(out_seq)
            in_seq = sequence.pad_sequences([in_seq], maxlen=max_wlen,padding='pre',value=0)[0]
            out_seq = np_utils.to_categorical([out_seq], num_classes=vocab_size+1)[0]
            x1.append(photo)
            x2.append(in_seq)
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
                      'words_input': np.array(x2)
                     }
            outputs = {'target_word':np.array(y),
                      'actual_sentence':np.array(sents)}
            yield inputs, outputs
            x1,x2,y,sents = [],[],[],[]


# In[312]:




with open('../dataset/train_images.p', 'rb') as filehandle:
    # read the data as binary data stream
    train_images_list = pickle.load(filehandle)
with open('../dataset/val_images.p', 'rb') as filehandle:
    # read the data as binary data stream
    val_images_list = pickle.load(filehandle)
with open('../dataset/test_images.p', 'rb') as filehandle:
    # read the data as binary data stream
    test_images_list = pickle.load(filehandle)

print(len(train_images_list)+len(val_images_list)+len(test_images_list))
train_batch_size = 64
val_batch_size = 64
train_gen = datagen(train_images_list,batch_size=train_batch_size)
val_gen = datagen(val_images_list,batch_size=val_batch_size)
test_gen = datagen(test_images_list,batch_size=val_batch_size)


# In[313]:


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


# In[314]:


# tx,ty = next(train_gen)
display(tx['image_input'].shape
        ,tx['words_input'][30]
        ,' '.join([co_to_w[x] if x !=0 else '' for x in tx['words_input'][30]])
        ,ty['actual_sentence'][1])


# In[315]:


len(val_images_list),len(train_images_list), w_to_co['xxxx']


# In[316]:


# # def _ctc_lambda_func(args):
# #     labels, y_pred, input_length, label_length = args
# #     return K.tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)
# vggnet.summary()
w_to_co['xxxx']


# In[317]:


print(co_to_w[1358])
print(embedding_matrix[1380])


# In[ ]:





# In[318]:


emb_out = emb.predict(tx['words_input'])
emb_out[1][-1]


# In[319]:


# parallel_model=multi_gpu_model(model, gpus=4)

# parallel_model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001,decay=1e-06),metrics=['accuracy']) #For multi GPU
# parallel_model.summary()
# # model_resnet.summary()
# # parallel_model.load_weights('./../checkpoints/sa_vgg16_3d_32x24_m32_direct.h5', by_name=True, skip_mismatch=True)


# In[320]:


# mc=ModelCheckpoint(filepath='../checkpoints/vgg16_lstm_words.h5',monitor='val_loss',period=1,save_best_only=True,save_weights_only=True,mode='auto',verbose=3)
# es=EarlyStopping(patience=300,monitor='val_loss',min_delta=0.0001,mode='auto')
print('training {0} for {1} epochs'.format(model_name,EPOCHS))


# In[322]:


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
window = 8
val_window = 128
train_iterations = len(train_images_list)//train_batch_size
val_iterations = len(val_images_list)//val_batch_size


min_v_l = 10
t_l, t_a, v_l, v_a = 0,0,0,0
for e in tqdm(np.arange(start=0,stop=EPOCHS),desc='Epoch'):
    start_time = time()
    tl,ta,vl,va = [],[],[],[]
    for im in tqdm(range(train_iterations),desc='Train_Iter',leave=False):
        tx,ty = next(train_gen)
        inputs1 = tx['image_input']
        inputs2 = tx['words_input']
        labels = ty['target_word']        
        for i in np.arange(len(inputs1),step=window):
            loss,accuracy = model.train_on_batch([inputs1[i:i+window],inputs2[i:i+window]],labels[i:i+window])
            tl.append(loss)
            ta.append(accuracy)
    for im in tqdm(range(val_iterations),desc='Val_Iter',leave=False):
        vx,vy = next(val_gen)
        inputs1 = vx['image_input']
        inputs2 = vx['words_input']
        labels = vy['target_word']
        for i in np.arange(len(inputs1),step=val_window):
            loss,accuracy = model.evaluate(verbose=0,x=[inputs1[i:i+val_window],inputs2[i:i+val_window]],y=labels[i:i+val_window])
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
    print('E:{0:3d}/{7}, tr_loss:{1:.4f}, tr_acc:{2:.4f}, v_loss:{3:.4f}, v_acc:{4:.4f} [{5:.2f} s/e] (min_v_loss:{6:.4f})'.format(e+1, t_l, t_a, v_l, v_a, np.mean(tt),min_v_l,EPOCHS))


# In[323]:


history = pd.DataFrame()
history['tr_acc'] = hist_ta
history['val_acc'] = hist_va
history['tr_loss'] = hist_tl
history['val_loss'] = hist_vl
display(history[['tr_acc','val_acc']].plot())
display(history[['tr_loss','val_loss']].plot())
history.to_csv('../tf_runs/{0}.csv'.format(model_name),index=False)


# # Evaluate

# In[324]:


model.load_weights('../checkpoints/{0}.h5'.format(model_name))


# In[325]:



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


# In[326]:


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


# In[329]:


vx,vy = next(val_gen)
photo = vx['image_input'][0]
plt.imshow(photo)
plt.show()
photo = np.expand_dims(photo,0)

print('Actual:',vy['actual_sentence'][0])
print()
st = time()
pred_greedy = predict_captions(photo)
et = time()
print('Greedy Predicted:{0},[{1:.2f} s]'.format(pred_greedy,et-st))


# In[ ]:



# st = time()
# pred_greedy = predict_captions(photo)
# et = time()
# print('Greedy Predicted:{0},[{1:.2f} s]'.format(pred_greedy,et-st))

st = time()
pred_bm3 = beam_search_predictions(photo,beam_index=3)
et = time()
print('Beam-3 Predicted:{0},[{1:.2f} s]'.format(pred_bm3,et-st))

st = time()
pred_bm5 = beam_search_predictions(photo,beam_index=5)
et = time()
print('Beam-5 Predicted:{0},[{1:.2f} s]'.format(pred_bm5,et-st))

# st = time()
# pred_bm7 = beam_search_predictions(photo,beam_index=7)
# et = time()
# print('Beam-7 Predicted:{0},[{1:.2f} s]'.format(pred_bm7,et-st))


# In[ ]:


from nltk.translate.bleu_score import sentence_bleu
reference = nlp(str(vy['actual_sentence'][0]))
reference = [[str(x) for x in list(reference)]]
df_result = pd.DataFrame()

candidate = nlp(pred_greedy)
candidate = [str(x) for x in list(candidate)]
df_result['greedy'] = [sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
                      ,sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
                      ,sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
                      ,sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))]

candidate = nlp(pred_bm3)
candidate = [str(x) for x in list(candidate)]
df_result['bm3'] = [sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
                      ,sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
                      ,sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
                      ,sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))]


candidate = nlp(pred_bm5)
candidate = [str(x) for x in list(candidate)]
df_result['bm5'] = [sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
                      ,sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
                      ,sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
                      ,sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))]


# candidate = nlp(pred_bm7)
# candidate = [str(x) for x in list(candidate)]
# df_result['bm7'] = [sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
#                       ,sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
#                       ,sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
#                       ,sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))]

df_result.index = ['BLEU-1','BLEU-2','BLEU-3','BLEU-4']
display(df_result.round(3).T)


# In[ ]:





# In[ ]:




