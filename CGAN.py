#####################################################################
################################ Conditional GAN ####################
#####################################################################

from keras.layers import Embedding
from keras.layers import Concatenate
import numpy as np
from numpy.random import randint
from numpy.random import randn
import tensorflow as tf
from keras.layers import Input,Conv2D,Dense,BatchNormalization,Conv2DTranspose,Flatten,Dropout,LeakyReLU,Reshape,ReLU
from keras.optimizers import Adam
from keras.models import Model,Sequential
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
######### 


#####
x_train = np.concatenate((x_train_re,x_train_le,x_train_n,x_train_l))
####
y_train_re = np.zeros((1,len(x_train_re)))
y_train_re = y_train_re[0]
y_train_le = np.ones((1,len(x_train_le)))
y_train_le = y_train_le[0]
y_train_n = 2*np.ones((1,len(x_train_n)))
y_train_n = y_train_n[0]
y_train_l = 3*np.ones((1,len(x_train_l)))
y_train_l = y_train_l[0]
#####
y_train = np.concatenate((y_train_re,y_train_le,y_train_n,y_train_l))
#####
y_train = y_train.astype('int8')
idx = np.random.randint(0,x_train.shape[0],x_train.shape[0])
x_train = x_train[idx]
y_train = y_train[idx]
x_train = x_train.astype('float32')
x_train = (x_train - 127.5) / 127.5

########## Define Descriminator

def discriminator(in_shape=(32,32,3), n_classes=4):
  in_label = Input(shape=(1,))
	# embedding for categorical input
  li = Embedding(n_classes, 50)(in_label)
	# scale up to image dimensions with linear activation
  n_nodes = in_shape[0] * in_shape[1]
  li = Dense(n_nodes)(li)
	# reshape to additional channel
  li = Reshape((in_shape[0], in_shape[1], 1))(li)
  input_layer = Input(shape=in_shape)
  merge = Concatenate()([input_layer, li])
  x = Conv2D(128 , (5,5) , padding ='same')(merge)
#  x = BatchNormalization()(x)
  x = LeakyReLU(alpha=0.2)(x)
#  x = Dropout(0.4)(x)
  x = Conv2D(128 , (5,5) , strides=(2,2) , padding ='same')(x)
#  x = BatchNormalization()(x)
  x = LeakyReLU(alpha=0.2)(x)
  x = Conv2D(128 , (5,5) , strides=(2,2) , padding ='same')(x)
#  x = BatchNormalization()(x)
  x = LeakyReLU(alpha=0.2)(x)
  x = Conv2D(256 , (5,5) , strides=(2,2) , padding ='same')(x)
#  x = BatchNormalization()(x)
  x = LeakyReLU(alpha=0.2)(x)
  x = Flatten()(x)
  x = Dropout(0.2)(x)
  final_layer = Dense(1 , activation='sigmoid')(x)
  dis_model = Model([input_layer,in_label],final_layer)
  opt = Adam(lr = 0.0002 , beta_1=0.5)
  dis_model.compile(loss = 'binary_crossentropy' , optimizer = opt , metrics = ['accuracy'])
  return dis_model

############ Define Generator

def generator(latent_dimension,n_classes=4):
  # label input
  in_label = Input(shape=(1,))
	# embedding for categorical input
  li = Embedding(n_classes, 50)(in_label)
  # linear multiplication
  li = Dense(4*4)(li)
	# reshape to additional channel
  li = Reshape((4, 4, 1))(li)
  input_layer = Input(shape=(latent_dimension,))
  x = Dense(4*4*128)(input_layer)
  x = LeakyReLU(alpha=0.2)(x)
  x = Reshape((4,4,128))(x)
  merge = Concatenate()([x, li])
  x = Conv2DTranspose(128 , (4,4) , strides=(2,2) , padding = 'same')(merge)
  x = LeakyReLU(alpha=0.2)(x)
  x = Conv2DTranspose(128 , (4,4) , strides=(2,2) , padding = 'same')(x)
  x = LeakyReLU(alpha=0.2)(x)
  x = Conv2DTranspose(256 , (4,4) , strides=(2,2) , padding = 'same')(x)
  x = LeakyReLU(alpha=0.2)(x)
  final_layer = Conv2D(3 ,(5,5), activation='tanh' , padding = 'same')(x)
  gen_model = Model([input_layer,in_label],final_layer)
  return gen_model

########## define gan 
def define_gan(g_model,d_model):
  gen_noise, gen_label = g_model.input
  gen_output = g_model.output
  gan_output = d_model([gen_output, gen_label])
  d_model.trainable = False
  gan_model = Model([gen_noise, gen_label],gan_output)
  opt = Adam(lr=0.0002 , beta_1=0.5)
  gan_model.compile(loss = 'binary_crossentropy',optimizer = opt)
  return gan_model

########### select real samples

def select_real_sample(dataset,n_sample):
  images, labels = dataset
  ix = randint(0,images.shape[0],n_sample)
  X, labels = images[ix], labels[ix]
  y = np.ones((n_sample,1))
  return [X,labels],y

########### generate latent points

def generate_latent_point(latent_dimension,n_samples,n_classes=4):
  x = randn(latent_dimension*n_samples)
  x = x.reshape(n_samples,latent_dimension)
  labels = randint(0, n_classes, n_samples)
  return [x, labels] 

########### generate fake sample

def generate_fake_samples(g_model,latent_dimension,n_sample):
  x , labels_input = generate_latent_point(latent_dimension,n_sample)
  x = g_model.predict([x , labels_input])
  y = np.zeros((n_sample,1))
  return [x,labels_input],y

########### save results

from matplotlib import pyplot as plt
def save_plot(examples,epoch):
  examples = (examples+1)/2
  for ii in range(10):
    plt.subplot(5,2,1+ii)
    plt.axis('off')
    plt.imshow(examples[ii])
  print(epoch)
  filename = '/home/user5.yazd/shell_script/result/cgan_ra_attack/result_e%03d.png' % (epoch+1)
  plt.savefig(filename)
  plt.close()

########## summarize

def summarize_performance(epoch , g_model,d_model,dataset,latent_dimension,n_sample=150):
  x_real,y_real = select_real_sample(dataset,n_sample)
  _ , acc_real = d_model.evaluate(x_real , y_real , verbose =0)
  x_fake , y_fake = generate_fake_samples(g_model,latent_dimension,n_sample)
  _ , acc_fake = d_model.evaluate(x_fake,y_fake,verbose=0)
  print('>>> ACC_real : %.0f%%, ACC_fake:%.0f%%' % (acc_real*100,acc_fake*100))
  save_plot(x_fake,epoch)
  file_name = '/content/drive/MyDrive/results/GAN/CGAN/cgan_ra_attack/generator_model_%03d.h5' % (epoch+1)
  g_model.save(file_name)

################ plot loss

def plot_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist):
	# plot loss
	plt.subplot(2, 1, 1)
	plt.plot(d1_hist, label='d-real')
	plt.plot(d2_hist, label='d-fake')
	plt.plot(g_hist, label='gen')
	plt.legend()
	# plot discriminator accuracy
	plt.subplot(2, 1, 2)
	plt.plot(a1_hist, label='acc-real')
	plt.plot(a2_hist, label='acc-fake')
	plt.legend()
	# save plot to file
	plt.savefig('/home/user5.yazd/shell_script/result/cgan_ra_attack/plot_line_plot_loss_cgan.png')
	plt.close()

#########################
def train(g_model,d_model,gan_model,dataset,latent_dimension,n_epoch,n_batch):
  bat_per_epo = int(len(dataset[0])/n_batch)
  half_batch = int(n_batch/2)
  for i in range(n_epoch):
    for j in range(bat_per_epo):
      [x_real,labels_real],y_real = select_real_sample(dataset,half_batch)
      d_loss1,d_acc1 = d_model.train_on_batch([x_real,labels_real], y_real)
      [x_fake,labels],y_fake = generate_fake_samples(g_model,latent_dimension,half_batch)
      d_loss2,d_acc2 = d_model.train_on_batch([x_fake,labels],y_fake)
      [x_gan,labels_input] = generate_latent_point(latent_dimension,n_batch)
      y_gan = np.ones((n_batch,1))
      g_loss = gan_model.train_on_batch([x_gan,labels_input],y_gan)
      print('>%d, %d/%d, d1=%.3f,d2=%.3f, g=%.3f' % (i+1, j+1, bat_per_epo, d_loss1,d_loss2, g_loss))
    save_plot(x_fake[:10],i)
      # record history
    if (i+1) % 1 == 0:
      d1_hist.append(d_loss1)
      d2_hist.append(d_loss2)
      g_hist.append(g_loss)
      a1_hist.append(d_acc1)
      a2_hist.append(d_acc2)
      file_name1 = '/home/user5.yazd/shell_script/result/cgan_ra_attack/generator_model_%03d.h5' % (i+1)
      g_model.save(file_name1)
      file_name2 = '/home/user5.yazd/shell_script/result/cgan_ra_attack/discriminator_%03d.h5' % (i+1)
      d_model.save(file_name2)
      #summarize_performance(i, g_model, d_model, dataset, latent_dimension)
  plot_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist)

############### final
d1_hist = []
d2_hist = []
g_hist = []
a1_hist = []
a2_hist = []
n_epoch = 101
n_batch = 128
latent_dimension = 100
d_model = discriminator()
g_model = generator(latent_dimension)
gan_model = define_gan(g_model,d_model)
dataset = [x_train,y_train]
train(g_model,d_model,gan_model,dataset,latent_dimension,n_epoch,n_batch)

