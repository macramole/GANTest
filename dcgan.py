from __future__ import print_function, division

#from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import numpy as np
import os
import csv

import dataset

class DCGAN():
    def __init__(self, outputPath, datasetDir = None, modelPath=None, img_rows = 128, img_cols = 128, channels = 3):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = channels
        self.datasetDir = datasetDir

        self.outputPath = outputPath
        self.modelsPath = os.path.join(self.outputPath, "models/")
        self.imagesPath = os.path.join(self.outputPath, "images/")
        self.logPath = os.path.join(self.outputPath,  "log.csv")

        if modelPath == None:
            os.makedirs( self.modelsPath, exist_ok=True )
            os.makedirs( self.imagesPath, exist_ok=True )

            optimizer = Adam(0.0002, 0.5)

            # Build and compile the discriminator
            self.discriminator = self.build_discriminator()

            self.discriminator.compile(loss='binary_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])

            # Build and compile the generator
            self.generator = self.build_generator()
            self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

            # The generator takes noise as input and generated imgs
            z = Input(shape=(100,))
            img = self.generator(z)

            # For the combined model we will only train the generator
            self.discriminator.trainable = False

            # The valid takes generated images as input and determines validity
            valid = self.discriminator(img)

            # The combined model  (stacked generator and discriminator) takes
            # noise as input => generates images => determines validity
            self.combined = Model(z, valid)
            self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        else:
            from keras.models import load_model

#            self.discriminator = load_model(modelPath + "discriminator.h5")
            self.generator = load_model(modelPath)
#            self.combined = load_model(modelPath + "combined.h5")

            from time import localtime, strftime
            self.outputPath = os.path.join(self.outputPath, strftime("%Y-%m-%d_%H-%M-%S", localtime()) )
            os.makedirs( self.outputPath , exist_ok=True )


    def build_generator(self):

        noise_shape = (100,)
        neurons = int(self.img_rows / 4)

        model = Sequential()

#        model.add(Dense(128 * 7 * 7, activation="relu", input_shape=noise_shape))
#        model.add(Reshape((7, 7, 128)))
        model.add(Dense(128 * neurons * neurons, activation="relu", input_shape=noise_shape))
        model.add(Reshape((neurons, neurons, 128)))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(512, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(256, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        img_shape = (self.img_rows, self.img_cols, self.channels)
        # img_shape = (self.channels, self.img_rows, self.img_cols)

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        # model.add(Dropout(0.1))

        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, img_save_interval=100, model_save_interval=200):

        try:
            # Load the dataset
            # (X_train, _), (_, _) = mnist.load_data()
            X_train = dataset.load_dataset(self.datasetDir, self.img_rows, self.img_cols)

            # Rescale -1 to 1
#            X_train = (X_train.astype(np.float32) - 127.5) / 127.5
            # X_train = np.expand_dims(X_train, axis=3)

            half_batch = int(batch_size / 2)

            for epoch in range(epochs):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random half batch of images
#                idx = np.random.randint(0, X_train.shape[0], half_batch)
                idx = np.arange(X_train.shape[0])
                np.random.shuffle(idx)
                idx = list(idx[0:half_batch])
                idx.sort()

                # idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = np.array(X_train[idx])
                imgs = (imgs.astype(np.float32) - 127.5) / 127.5

                # Sample noise and generate a half batch of new images
                noise = np.random.normal(0, 1, (half_batch, 100))
                gen_imgs = self.generator.predict(noise)
                # print(imgs.shape)

                # Train the discriminator (real classified as ones and generated as zeros)
                d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))

                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Train Generator
                # ---------------------

                noise = np.random.normal(0, 1, (batch_size, 100))

                # Train the generator (wants discriminator to mistake images as real)
                g_loss = self.combined.train_on_batch(noise, np.ones((batch_size, 1)))

                # Plot the progress
                print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

                with open(self.logPath, "a") as logfile:
                    logwriter = csv.writer(logfile)
                    logwriter.writerow([epoch, d_loss[0], 100*d_loss[1], g_loss])

                # If at save interval => save generated image samples
                if epoch % img_save_interval == 0:
                    self.generate(epoch)
                if epoch % model_save_interval == 0:
                    self.combined.save(os.path.join(self.modelsPath, "combined.%d.h5" % epoch))
                    self.generator.save(os.path.join(self.modelsPath, "generator.%d.h5" % epoch))
                    self.discriminator.save(os.path.join(self.modelsPath, "discriminator.%d.h5" % epoch))

            self.combined.save(os.path.join(self.modelsPath, "combined.%d.h5" % epochs))
            self.generator.save(os.path.join(self.modelsPath, "generator.%d.h5" % epochs))
            self.discriminator.save(os.path.join(self.modelsPath, "discriminator.%d.h5" % epochs))

        except KeyboardInterrupt:
            self.combined.save(os.path.join(self.modelsPath, "combined.unfinished.h5"))
            self.generator.save(os.path.join(self.modelsPath, "generator.unfinished.h5"))
            self.discriminator.save(os.path.join(self.modelsPath, "discriminator.unfinished.h5"))


    def generate(self, epoch = None):
#         r, c = 5, 5
#         noise = np.random.normal(0, 1, (r * c, 100))
#         gen_imgs = self.generator.predict(noise)
# #        print(gen_imgs.shape)
#
#         # Rescale images 0 - 1
#         gen_imgs = 0.5 * gen_imgs + 0.5

#        just testing if colors are ok
#
#        from keras.preprocessing.image import load_img, img_to_array
#        img = load_img("instagram_manoloide/2016_05_09_12_51_13113803_869336679859788_647270116_n.jpg")
#        img.thumbnail((28,28))
#        img = img_to_array(img)
#        img = (img.astype(np.float32) - 127.5) / 127.5
#        img = 0.5 * img + 0.5
#
#        img = img.reshape(1,28,28,3)
#        print(img.shape)
#
#        gen_imgs = img



        if epoch != None:
            r, c = 3, 3
            noise = np.random.normal(0, 1, (r * c, 100))
            gen_imgs = self.generator.predict(noise)
            # print(gen_imgs.shape)
            # gen_imgs = gen_imgs.reshape(r*c,self.img_rows,self.img_cols)

            # Rescale images 0 - 1
            gen_imgs = 0.5 * gen_imgs + 0.5

            fig = plt.figure(1,(9,9))
            grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(r, c),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

            for i in range(r*c):
                # grid[i].imshow( gen_imgs[i], cmap='gray')
                grid[i].imshow( gen_imgs[i])

            fig.savefig( os.path.join(self.imagesPath, "out_%04d.png" % epoch) )
        else :
            import scipy.misc

#            cantImgs = 500
#            noise = np.random.normal(0, 1, (cantImgs, 100))
#            gen_imgs = self.generator.predict(noise)
#            gen_imgs = 0.5 * gen_imgs + 0.5
#
#            for i in range(cantImgs):
#                scipy.misc.imsave('dcgan/images/final/%d.jpg' % i, gen_imgs[i, :, :, :])

#            noise = np.random.normal(0, 1, (1, 100))
#            noise = np.zeros((1, 100))

#            cnt = 0
#            for i in np.arange(0,1.1,0.01):
#                noise = np.repeat(i, 100)
#                noise = noise.reshape((1,100))
#    #            noise[0,i] = 0
#                gen_imgs = dcgan.generator.predict(noise)
#                gen_imgs = 0.5 * gen_imgs + 0.5
#
##                axs = plt.subplot()
##                fig = axs.figure
##                axs.imshow( gen_imgs[0, :,:,:] )
##                axs.axis("off")
##
#
#                scipy.misc.imsave('dcgan/images/video3/%d.jpg' % cnt, gen_imgs[0, :, :, :])
#                cnt+=1


#            noise = np.repeat(0.0, 100)
#            noise = noise.reshape((1,100))
#            cnt = 0
#            for x in range(100):
#                for i in np.arange(0,1.1,0.5):
##                    noise = np.repeat(i, 100)
##                    noise = noise.reshape((1,100))
#                    noise[0,x] = i
##                    noise[0,0] = i
#                    gen_imgs = dcgan.generator.predict(noise)
#                    gen_imgs = 0.5 * gen_imgs + 0.5
#
#    #                axs = plt.subplot()
#    #                fig = axs.figure
#    #                axs.imshow( gen_imgs[0, :,:,:] )
#    #                axs.axis("off")
#    #
#
#                    scipy.misc.imsave('dcgan/images/video2/%d.png' % cnt, gen_imgs[0, :, :, :])
#                    cnt += 1

            # Sample and interpolate

            from scipy.interpolate import interp1d

            where = os.path.join( self.outputPath, "%05d.png" ) #'dcgan/images/4_wiki-women_2/video1/%05d.png'
            cantRandom = 40
            cantInterpolation = 30

            realNoise = np.random.normal(0, 1, (cantRandom-1, 100))
            realNoise = np.vstack( ( realNoise, realNoise[0] ) )

            cnt = 0
            for i in range(cantRandom-1):
                x = np.array([0,1])
                y = realNoise[i:i+2]
                f = interp1d( x , y, axis = 0  )

                noise = f( np.linspace(0,1,cantInterpolation+1, endpoint = False) )
                gen_imgs = self.generator.predict(noise)
                gen_imgs = 0.5 * gen_imgs + 0.5

                for j in range(cantInterpolation+1):
                    scipy.misc.imsave(where % cnt, gen_imgs[j])
                    cnt +=1

            # Big test

#            model = dcgan.generator.get_layer(index=1)
#            model.add(UpSampling2D(size=(5,5), name = "superup" ))
#            model.add( Dense(1000) )
#            model.add( Reshape( (None,None,3) ) )
#            model.compile
#            model.summary()
#
#            model.compile(loss='binary_crossentropy', optimizer=optimizer)
#
#            realNoise = np.random.normal(0, 1, (1, 100))
#            gen_imgs = model.predict(realNoise)
#            gen_imgs = 0.5 * gen_imgs + 0.5
#
#            scipy.misc.imsave('dcgan/images/out-big.png', gen_imgs[0])

#if __name__ == '__main__':
#     try:
#         dcgan = DCGAN()
#
#         with open("dcgan/saved_model/log.csv", "w") as logfile:
#             logwriter = csv.writer(logfile)
#             logwriter.writerow(["epoch","d_loss", "d_acc", "g_loss"])
#
#         dcgan.train(epochs=40000, batch_size=32, save_interval=100)
#
#         dcgan.combined.save("dcgan/saved_model/combined.h5")
#         dcgan.generator.save("dcgan/saved_model/generator.h5")
#         dcgan.discriminator.save("dcgan/saved_model/discriminator.h5")
#     except KeyboardInterrupt:
#         dcgan.combined.save("dcgan/saved_model/combined.unfinished.h5")
#         dcgan.generator.save("dcgan/saved_model/generator.unfinished.h5")
#         dcgan.discriminator.save("dcgan/saved_model/discriminator.unfinished.h5")


#    dcgan = DCGAN("dcgan/saved_model/wiki-women/")
#    dcgan.generate()
