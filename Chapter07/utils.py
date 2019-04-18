import numpy as np
from PIL import Image
from glob import glob
from keras.models import model_from_json


def load_model(filepath_model, filepath_weights):
    # load json and create model
    json_file = open(filepath_model, 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    # load weights into new model
    model.load_weights(filepath_weights)
    print("Loaded model from disk")
    return model


def save_model(model, filepath_model, filepath_weights):
    # serialize model to JSON
    model_json = model.to_json()
    with open(filepath_model, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(filepath_weights)
    print("Saved model to disk")


def load_image(filepath, res=None, scale=True):
    image = Image.open(filepath)
    if res is not None:
        image = image.resize([res, res], Image.BILINEAR)

    image = np.array(image, dtype=int)
    if scale:
        image = 2 * (image / 255.0) - 1
        image = image.astype(float)
    return image


def iterate_minibatches(glob_str, batch_size=128, res=128):
    if glob_str == 'cifar10':
        data = get_cifar10_data()
        i = 0
        n_samples = len(data)
        while True:
            # drop last?
            if len(data) - i*batch_size < batch_size or i == 0:
                ids = np.random.randint(0, n_samples, n_samples)
                np.random.shuffle(ids)
                i = 0

            imgs = data[ids[i*batch_size:(i+1)*batch_size]]

            i = (i + 1) % len(data)
            yield imgs
    else:
        filepaths = glob(glob_str)
        n_files = len(filepaths)
        cur_batch = 0
        while True:
            # drop last if it does not fit
            if (n_files - cur_batch*batch_size) < batch_size or cur_batch == 0:
                ids = np.random.randint(0, n_files, n_files)
                np.random.shuffle(ids)
                cur_batch = 0

            train_data = []
            for i in range(batch_size):
                image = load_image(filepaths[ids[cur_batch*batch_size+i]], res)
                train_data.append(image)

            train_data = np.array(train_data)
            yield train_data


def get_cifar10_data():
    from keras.datasets import cifar10
    # load cifar10 data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # convert train and test data to float32
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # scale train and test data to [-1, 1]
    X_train = (X_train / 255) * 2 - 1
    X_test = (X_train / 255) * 2 - 1

    return X_train
    # return (X_train, y_train), (X_test, y_test)


def log_images(images, name, i, logger, h=256, w=256, shape=6):
    images = (images.reshape(shape, shape, h, w, images.shape[-1])
                    .transpose(0, 2, 1, 3, 4)
                    .reshape(shape*h, shape*w, images.shape[-1]))
    images = ((images + 1) * 0.5)
    logger.add_image('{}'.format(name, i), images, i, dataformats='HWC')


def log_losses(loss_d, loss_g, iteration, logger):
    names = ['loss_d', 'loss_d_real_fake', 'loss_d_gp', 'loss_d_real_drift']
    for i in range(len(loss_d)):
        logger.add_scalar(names[i], loss_d[i], iteration)
    logger.add_scalar("losses_g", loss_g, iteration)
