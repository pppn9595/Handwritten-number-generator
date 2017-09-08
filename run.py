import os, matplotlib.pyplot as plt, numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.datasets import mnist
from keras.layers import *
from keras.models import *
from keras.optimizers import *

class GAN():
    def __init__(self):
        self.noise_shape = (100,)
        self.img_shape = (28, 28, 1)
        self.build_model()

    def build_generator(self):
        return Sequential([
            Dense(256, input_shape=self.noise_shape),
            LeakyReLU(alpha=0.2),
            Dense(512),
            LeakyReLU(alpha=0.2),
            Dense(1024),
            LeakyReLU(alpha=0.2),
            Dense(np.prod(self.img_shape), activation='tanh'),
            Reshape(self.img_shape),
        ])

    def build_discriminator(self):
        return Sequential([
            Flatten(input_shape=self.img_shape),
            Dense(512),
            LeakyReLU(alpha=0.2),
            Dense(256),
            LeakyReLU(alpha=0.2),
            Dense(1, activation='sigmoid'),
        ])

    def build_model(self):
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        self.discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
        self.generator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

        self.discriminator.trainable = False

        noise = Input(shape=self.noise_shape)
        img = self.generator(noise)
        valid = self.discriminator(img)

        self.combined = Model(inputs=noise, outputs=valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))



    def train(self, batch_size=128, save_interval=200):
        X_train = mnist.load_data()[0][0]

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        zeros = np.zeros((batch_size, 1))
        ones = np.ones((batch_size, 1))

        epoch = 1
        while True:

            # Train Discriminator
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, 100))

            gen_imgs = self.generator.predict(noise)

            self.discriminator.train_on_batch(imgs, ones)
            self.discriminator.train_on_batch(gen_imgs, zeros)


            # Train Generator
            noise = np.random.normal(0, 1, (batch_size, 100))

            self.combined.train_on_batch(noise, ones)

            if epoch % save_interval == 0:
                self.save_imgs(epoch)
                print('Epoch: %d | Image saved' % epoch)

            epoch += 1

    def save_imgs(self, epoch, directory='images'):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = self.generator.predict(noise)
        gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        if not os.path.exists(directory):
            os.makedirs(directory)
        fig.savefig("%s/epoch_%d.png" % (directory, epoch))
        plt.close()


gan = GAN()
gan.train()