import io
from PIL import Image
import numpy as np
import h5py
from keras.models import model_from_json
from glob import glob


def load_model(folder_path):
    # load json and create model
    json_file = open("{}/model.json".format(folder_path), 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    # load weights into new model
    model.load_weights("{}/model.h5".format(folder_path))
    print("Loaded model from disk")
    return model


def save_model(model, folder_path):
    # serialize model to JSON
    model_json = model.to_json()
    with open("{}/model.json".format(folder_path), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("{}/model.h5".format(folder_path))
    print("Saved model to disk")


def load_image(filepath, scale=True):
    images = np.array(Image.open(filepath), dtype=int)
    if scale:
        images = 2 * (images / 255.0) - 1
        images = images.astype(float)
    return images


def iterate_minibatches(glob_str, batch_size=128, img_size=256):
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
            image_ab = load_image(filepaths[ids[cur_batch*batch_size+i]])
            train_data.append([image_ab[:, :img_size], image_ab[:, img_size:]])

        cur_batch = (cur_batch + 1) % int(len(filepaths)/batch_size)

        train_data = np.array(train_data)
        yield train_data, cur_batch


def get_edges(t):
    edge = np.zeros(t.shape, dtype=bool)
    edge[:,1:]  = edge[:,1:]  | (t[:,1:] != t[:,:-1])
    edge[:,:-1] = edge[:,:-1] | (t[:,1:] != t[:,:-1])
    edge[1:,:]  = edge[1:,:]  | (t[1:,:] != t[:-1,:])
    edge[:-1,:] = edge[:-1,:] | (t[1:,:] != t[:-1,:])
    return edge.astype(float)


def iterate_minibatches_cityscapes(folder_path, prefix='train', use_edges=True,
                                   batch_size=3):
    filepaths_img = sorted(
        glob("{}/{}_img/*.png".format(folder_path, prefix)))
    filepaths_label = sorted(
        glob("{}/{}_label/*.png".format(folder_path, prefix)))
    if use_edges:
        filepaths_inst = sorted(
            glob("{}/{}_inst/*.png".format(folder_path, prefix)))

    n_files = len(filepaths_img)
    cur_batch = 0
    while True:
        # drop last if it does not fit
        if (n_files - cur_batch*batch_size) < batch_size or cur_batch == 0:
            ids = np.random.choice(range(n_files), n_files, replace=False)
            np.random.shuffle(ids)
            cur_batch = 0
        train_data = []
        for i in range(batch_size):
            img = load_image(filepaths_img[ids[cur_batch*batch_size+i]])
            lbl = load_image(filepaths_label[ids[cur_batch*batch_size+i]])
            lbl = lbl[:, :, None]
            if use_edges:
                inst = load_image(filepaths_inst[ids[cur_batch*batch_size+i]])
                edges = get_edges(inst)[..., None]
                train_data.append(np.dstack((img, lbl, edges)))
            else:
                train_data.append(np.dstack((img, lbl)))

        cur_batch = (cur_batch + 1) % int(len(filepaths_img)/batch_size)

        yield np.array(train_data), cur_batch


def log_images(images, name, i, logger, h=256, w=256, shape=4):
    images = (images.reshape(shape, shape, h, w, images.shape[-1])
                    .transpose(0, 2, 1, 3, 4)
                    .reshape(shape*h, shape*w, images.shape[-1]))
    images = ((images + 1) * 0.5)
    logger.add_image('{}'.format(name, i), images, i, dataformats='HWC')


def log_losses(loss_d, loss_g, iteration, logger):
    logger.add_scalar("loss_d", loss_d[0], iteration)
    logger.add_scalar("loss_d_real", loss_d[1], iteration)
    logger.add_scalar("loss_d_fake", loss_d[2], iteration)
    logger.add_scalar("loss_g", loss_g[0], iteration)
    logger.add_scalar("loss_g_fake", loss_g[1], iteration)
    logger.add_scalar("loss_g_reconstruction", loss_g[2], iteration)


def log_losses_pix2pixhd(loss_d, loss_g, iteration, logger):
    for i in range(len(loss_d)):
        logger.add_scalar("loss_d{}".format(i), loss_d[i][0], iteration)
        logger.add_scalar("loss_d{}_real".format(i), loss_d[i][1], iteration)
        logger.add_scalar("loss_d{}_fake".format(i), loss_d[i][2], iteration)
    logger.add_scalar("loss_g", loss_g[0], iteration)
    logger.add_scalar("loss_g_fake", loss_g[1], iteration)
    logger.add_scalar("loss_g_reconstruction", loss_g[2], iteration)
