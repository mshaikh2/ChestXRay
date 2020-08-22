import keras.layers as kl
import AttentionMed as AM
import numpy as np

# from keras.models import Model



class ImageEncoder:
    def model(image_input):
#         image_input = kl.Input(shape=(256,256,3),name='image_input')
        x = kl.Conv2D(filters=64, kernel_size=(3,3), padding='same', name='block1_conv1')(image_input)    
        x = kl.Conv2D(filters=64, kernel_size=(3,3), padding='same', name='block1_conv2')(x)    
        x = kl.MaxPool2D(pool_size=2, strides=2, padding='same', name='block1_reduction_pool')(x)
        x = kl.BatchNormalization()(x)
        x = kl.Activation('relu')(x)

    #     x = kl.Dropout(img_dropout_rate)(x)

        x = kl.Conv2D(filters=128, kernel_size=(3,3), padding='same', name='block2_conv1')(x)
        x = kl.Conv2D(filters=128, kernel_size=(3,3), padding='same', name='block2_conv2')(x)
        x = kl.MaxPool2D(pool_size=2, strides=2, padding='same', name='block2_reduction_pool')(x)
        x = kl.BatchNormalization()(x)
        x = kl.Activation('relu')(x)
    #     x = kl.Dropout(img_dropout_rate)(x)

        x = kl.Conv2D(filters=256, kernel_size=(3,3), padding='same', name='block3_conv1')(x)
        x = kl.Conv2D(filters=256, kernel_size=(3,3), padding='same', name='block3_conv2')(x)
        x = kl.Conv2D(filters=256, kernel_size=(3,3), padding='same', name='block3_conv3')(x)
        x = kl.MaxPool2D(pool_size=2, strides=2, padding='same', name='block3_reduction_pool')(x)
        x = kl.BatchNormalization()(x)
        x = kl.Activation('relu')(x)

    #     x = kl.Dropout(img_dropout_rate)(x)

        x = kl.Conv2D(filters=512, kernel_size=(3,3), padding='same', name='block4_conv1')(x)
        x = kl.Conv2D(filters=512, kernel_size=(3,3), padding='same', name='block4_conv2')(x)
        x = kl.Conv2D(filters=512, kernel_size=(3,3), padding='same', name='block4_conv3')(x)
        x = kl.MaxPool2D(pool_size=2, strides=2, padding='same', name='block4_reduction_pool')(x)
        x = kl.BatchNormalization()(x)
        x = kl.Activation('relu')(x)

    #     x = kl.Dropout(img_dropout_rate)(x)

        x = kl.Conv2D(filters=512, kernel_size=(3,3), padding='same', name='block5_conv1')(x)
        x = kl.Conv2D(filters=512, kernel_size=(3,3), padding='same', name='block5_conv2')(x)
        x = kl.Conv2D(filters=512, kernel_size=(3,3), padding='same', name='block5_conv3')(x)
    #     x, amaps = SoftAttention(ch=512, m=32, name='soft_attention')(x)
        x = kl.MaxPool2D(pool_size=2, strides=2, padding='same', name='block5_reduction_pool')(x)
        x = kl.BatchNormalization()(x)
        x = kl.Activation('relu',name='image_output_features')(x)
        reshaped = (int(x.shape[1])*int(x.shape[2]),int(x.shape[-1]))
        encoded_image = kl.Lambda(lambda x: kl.Reshape(target_shape=reshaped)(x),name='hw_flat_image_output_features')(x)
        
        return encoded_image#Model(image_input,encoded_image)
    
class TextDecoder:
    def model(embedding_size,vocab_size,encoded_image,words_input,masks,positional_encoding,bottleneck_units=512):
        
        emb = kl.Embedding(vocab_size+1, embedding_size, mask_zero=False, name='w2v_emb')(words_input)
        emb = kl.Conv1D(name='c1',filters=bottleneck_units,activation='relu',strides=1,kernel_size=1,padding='same')(emb)
        sa_input = kl.Add()([emb,positional_encoding])

        # ---- decoder block 1 ----
        dec_1 = decoder_block(sa_input=sa_input
                             ,masks=None
                             ,encoder_output=encoded_image
                             ,heads=8
                             ,layer_number=1)

        # ---- decoder block 2 ----
        dec_2 = decoder_block(sa_input=dec_1
                             ,masks=None
                             ,encoder_output=encoded_image
                             ,heads=8
                             ,layer_number=2)

        gap = kl.GlobalAveragePooling1D()(dec_2)
        gap = kl.Dense(1024,activation='relu')(gap)
        target = kl.Dense(vocab_size+1)(gap)
        
        return target
    

def multiheadSelfAttention(prev_layer,masks,layer_number=0,heads=8):
    assert prev_layer != None
    sa_arr = []
    for head in range(heads):
        if masks!=None:
            x,q,k,v,s,b,beta,o = AM.SelfAttention(ch=int(prev_layer.shape[-1]),name='sa{0}{1}'.format(layer_number,head))([prev_layer,masks])
        else:
            x,q,k,v,s,b,beta,o = AM.SelfAttention(ch=int(prev_layer.shape[-1]),name='sa{0}{1}'.format(layer_number,head))(prev_layer)
#         print(q.shape)
#         print(k.shape)
#         print(v.shape)
#         print(s.shape)
#         print(b.shape)
#         print(beta.shape)
#         print(o.shape)
        sa = kl.BatchNormalization()(o)
        sa_arr.append(o)
    return sa_arr

def condResAndNormSelfAttn(sa_layer,residual_inp,out_channels,attn_type,layer_number=0,condense=False,method='add'):
    assert attn_type!=None
    if condense:
        x = kl.Concatenate(name='concat_{0}_{1}'.format(attn_type,layer_number))(sa_layer)
        x = AM.CondenseAttention1D(ch_in=int(x.shape[-1]),name='cond_{0}_{1}'.format(attn_type,layer_number),ch_out=out_channels)(x)
    else:
        x = sa_layer
    x, g1, g2 = AM.ResidualCombine(method=method
                                   ,name='res_comb_{0}_{1}'.format(attn_type,layer_number))([residual_inp, x])
#     x = kl.Conv1D(filters=512,activation='relu',strides=1,kernel_size=1,name='ff_conv_{0}'.format(layer_number))(x)
    x = kl.BatchNormalization()(x)
    return x

def enc_dec_attn(encoder,decoder,out_channels,masks=None,layer_number=0,heads=8):
    assert encoder != None
    assert decoder != None
    eca_arr = []
    for head in range(heads):
        q,k,v,beta,scores,o = AM.EncDecAttention(enc_ch=int(encoder.shape[-1]),dec_ch=int(decoder.shape[-1]),name='eca{0}{1}'.format(layer_number,head))([decoder,encoder])
        eca = kl.BatchNormalization()(o)
        eca_arr.append(eca)
    return eca_arr

def multiheadAttention(prev_layer,context_vector,layer_number=0,heads=8):
    assert prev_layer != None
    assert context_vector != None
    a_arr = []
    for head in range(heads):
        a,a_map = AM.Attention(ch=int(prev_layer.shape[-1]),name='a{0}{1}'.format(layer_number,head))([context_vector,prev_layer])
        a = kl.BatchNormalization()(a)
        a_arr.append(a)
    return a_arr

def condResAndNormAttn(a_layer,context_vector,out_channels,layer_number=0,method='add'):
    x = kl.Concatenate(name='concat_attn_{0}'.format(layer_number))(a_layer)
    x = AM.CondenseAttention1D(ch_in=int(x.shape[-1]),name='condense_attn1D_{0}'.format(layer_number),ch_out=out_channels)(x)
    x = kl.Lambda(lambda x:K.squeeze(x,axis=1),name="squeeze_attn_{0}".format(layer_number))(x)
    x, g1, g2 = AM.ResidualCombine(method=method
                                   ,name='residual_combine_attn_{0}'.format(layer_number))([context_vector, x])
    x = kl.BatchNormalization()(x)
    return x

def decoder_block(sa_input,encoder_output,masks=None,layer_number=1,heads=8):
    mhsa = multiheadSelfAttention(prev_layer=sa_input,masks=masks,layer_number=layer_number,heads=heads)
    # add+norm
    add_norm_mhsa = condResAndNormSelfAttn(attn_type='sa',condense=True,sa_layer=mhsa,residual_inp=sa_input,out_channels=int(sa_input.shape[-1]),layer_number=layer_number)
    # enc-dec attn
    eca = enc_dec_attn(encoder=encoder_output,decoder=add_norm_mhsa,masks=masks,out_channels=int(sa_input.shape[-1]),layer_number=layer_number) 
    # add+norm
    add_norm_eca = condResAndNormSelfAttn(attn_type='enc',condense=True,sa_layer=eca,residual_inp=add_norm_mhsa,out_channels=int(sa_input.shape[-1]),layer_number=layer_number)
    # apply same feed forward on each position
    ff_dec = kl.TimeDistributed(kl.Dense(512,activation='relu'))(add_norm_eca)
    # add+norm
    add_norm_eca = condResAndNormSelfAttn(attn_type='NA',condense=False,sa_layer=ff_dec,residual_inp=add_norm_eca,out_channels=int(sa_input.shape[-1]),layer_number=layer_number)
    return add_norm_eca

def get_angles(pos, i, dimensions):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(dimensions))
    return pos * angle_rates

def positional_encoding(position, dimensions):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(dimensions)[np.newaxis, :],
                          dimensions)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return pos_encoding