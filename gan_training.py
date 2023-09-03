from numpy import zeros
from numpy import ones
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from matplotlib import pyplot
import numpy as np
import argparse
import os

from title_dataset_one_hot import TitleDatasetOneHot


# define the discriminator model
def define_discriminator(image_shape):
    print("===> define_discriminator", image_shape)
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # source image input
    in_src_image = Input(shape=(256, 256, 16))
    # target image input
    in_target_image = Input(shape=(256, 256, 4))
    # concatenate images channel-wise
    merged = Concatenate()([in_src_image, in_target_image])
    # C64
    d = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)
    # C128
    d = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256
    d = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512
    d = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # second last output layer
    d = Conv2D(512, (4, 4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # patch output
    d = Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)
    # define model
    model = Model([in_src_image, in_target_image], patch_out)
    # compile model
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    return model


# define an encoder block
def define_encoder_block(layer_in, n_filters, batchnorm=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add downsampling layer
    g = Conv2D(n_filters, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(layer_in)
    g = Conv2D(n_filters, (2, 2), strides=(2, 2), padding='valid', kernel_initializer=init)(g)
    # conditionally add batch normalization
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    # leaky relu activation
    g = LeakyReLU(alpha=0.2)(g)
    return g


# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add upsampling layer
    g = Conv2DTranspose(n_filters, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(layer_in)
    g = Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='valid', kernel_initializer=init)(g)
    # add batch normalization
    g = BatchNormalization()(g, training=True)
    # conditionally add dropout
    if dropout:
        g = Dropout(0.5)(g, training=True)
    # merge with skip connection
    g = Concatenate()([g, skip_in])
    # relu activation
    g = Activation('relu')(g)
    return g


# define the standalone generator model
def define_generator(image_shape=(256, 256, 16)):
    print("===> define_generator", image_shape)
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=(256, 256, 16))
    # encoder model
    e1 = define_encoder_block(in_image, 64, batchnorm=False)
    e2 = define_encoder_block(e1, 128)
    e3 = define_encoder_block(e2, 256)
    e4 = define_encoder_block(e3, 512)
    e5 = define_encoder_block(e4, 512)
    e6 = define_encoder_block(e5, 512)
    e7 = define_encoder_block(e6, 512)
    # bottleneck, no batch norm and relu
    b = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(e7)
    b = Activation('relu')(b)
    # decoder model
    d1 = decoder_block(b, e7, 512)
    d2 = decoder_block(d1, e6, 512)
    d3 = decoder_block(d2, e5, 512)
    d4 = decoder_block(d3, e4, 512, dropout=False)
    d5 = decoder_block(d4, e3, 256, dropout=False)
    d6 = decoder_block(d5, e2, 128, dropout=False)
    d7 = decoder_block(d6, e1, 64, dropout=False)
    # output
    g = Conv2DTranspose(4, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d7)
    out_image = Activation('tanh')(g)
    # define model
    model = Model(in_image, out_image)
    return model


# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape):
    print("===> define_gan", image_shape)
    # make weights in the discriminator not trainable
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    # define the source image
    in_src = Input(shape=(256, 256, 16))
    # connect the source image to the generator input
    gen_out = g_model(in_src)
    # connect the source input and generator output to the discriminator input
    dis_out = d_model([in_src, gen_out])
    # src image as input, generated image and classification output
    model = Model(in_src, [dis_out, gen_out])
    # compile model
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1, 100])
    return model


# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
    # unpack dataset
    # trainA, trainB = dataset
    # choose random instances
    ix = randint(0, len(dataset), n_samples)
    # retrieve selected images
    X1 = []
    X2 = []
    for i in ix:
        data, label = dataset[i]
        X1.append(label)
        X2.append(data)
    X1 = np.array(X1)
    X2 = np.array(X2)

    # X1, X2 = trainA[ix], trainB[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, patch_shape, patch_shape, 1))
    return [X1, X2], y


# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
    # generate fake instance
    X = g_model.predict(samples)
    # create 'fake' class labels (0)
    y = zeros((len(X), patch_shape, patch_shape, 1))
    return X, y


# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, dataset, work_dir, n_samples=3):
    try:
        # select a sample of input images
        [X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
        # generate a batch of fake samples
        X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
        # scale all pixels from [-1,1] to [0,1]

        print(X_realA.shape, X_realB.shape, X_fakeB.shape)

        X_realA = np.argmax(X_realA, axis=-1)

        X_realB = X_realB[:, :, :, :-1] + 0.5
        X_fakeB = X_fakeB[:, :, :, :-1] + 0.5
        X_realB[X_realB < 0] = 0
        X_fakeB[X_fakeB < 0] = 0

        norm_value = X_realB.max()
        X_realB = (255 * X_realB / norm_value).astype(np.uint8)
        X_fakeB = (255 * X_fakeB / norm_value).astype(np.uint8)

        # plot real source images
        for i in range(n_samples):
            pyplot.subplot(3, n_samples, 1 + i)
            pyplot.axis('off')
            pyplot.imshow(X_realA[i])
        # plot generated target image
        for i in range(n_samples):
            pyplot.subplot(3, n_samples, 1 + n_samples + i)
            pyplot.axis('off')
            pyplot.imshow(X_fakeB[i])
        # plot real target image
        for i in range(n_samples):
            pyplot.subplot(3, n_samples, 1 + n_samples * 2 + i)
            pyplot.axis('off')
            pyplot.imshow(X_realB[i])
        # save plot to file
        filename1 = 'plot_model_gan_%06d.png' % (step + 1)
        filename1 = os.path.join(work_dir, filename1)
        pyplot.savefig(filename1)
        pyplot.close()
    except:
        filename1 = 'plot_model_gan_%06d.png' % (step + 1)
        filename1 = os.path.join(work_dir, filename1)
    # save the generator model
    filename2 = 'model_gan_%06d.h5' % (step + 1)
    filename2 = os.path.join(work_dir, filename2)
    g_model.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))


# train pix2pix models
def train(d_model, g_model, gan_model, dataset, work_dir, n_epochs=30, n_batch=32):
    # determine the output square shape of the discriminator
    n_patch = d_model.output_shape[1]
    # unpack dataset
    # trainA, trainB = dataset
    # calculate the number of batches per training epoch
    bat_per_epo = 40  # int(len(trainA) / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # manually enumerate epochs
    for i in range(n_steps):
        # select a batch of real samples
        [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
        # generate a batch of fake samples
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        # update discriminator for real samples
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        # update discriminator for generated samples
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        # update the generator
        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        # summarize performance
        print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i + 1, d_loss1, d_loss2, g_loss))
        # summarize model performance
        if i % bat_per_epo == 0:
            summarize_performance(i, g_model, dataset, work_dir)


# load image data
def get_args():
    """Argument parser.
    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', type=str, required=True,
                        help='location for writing checkpoints and exporting models')
    args, _ = parser.parse_known_args()
    return args


def main(args):
    work_dir = args.job_dir

    print("===> Loading Dataset")
    npzfile = np.load(os.path.join(work_dir, "train_data_real.npz"))
    dataset = TitleDatasetOneHot(npzfile, {}, is_train=True)

    print("===> Model definition")
    # define input shape based on the loaded dataset
    image_shape1 = dataset[0][1].shape
    image_shape2 = dataset[0][0].shape
    # define the models
    d_model = define_discriminator(image_shape2)
    g_model = define_generator(image_shape1)
    # define the composite model
    gan_model = define_gan(g_model, d_model, image_shape2)
    # train model
    print("===> Training")
    train(d_model, g_model, gan_model, dataset, work_dir)


main(get_args())
