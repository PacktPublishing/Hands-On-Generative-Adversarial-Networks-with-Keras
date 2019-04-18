import io
from PIL import Image
import numpy as np
import h5py


def images_from_bytes(byte_images, img_size=(64, 64)):
    images = [
        np.array(Image.open(io.BytesIO(img)).resize(img_size), dtype=int)
        for img in byte_images]
    images = np.array(images)
    images = 2 * (images / 255.0) - 1
    images = images.astype(float)
    return images


def iterate_minibatches(data, batch_size=128):
    i = 0
    n_samples = len(data[0])
    while True:
        # drop last?
        if len(data) - i*batch_size < batch_size or i == 0:
            ids = np.random.randint(0, n_samples, n_samples)
            np.random.shuffle(ids)
            i = 0

        imgs = data[0][ids[i*batch_size:(i+1)*batch_size]]
        embs = data[1][ids[i*batch_size:(i+1)*batch_size]]
        txts = data[2][ids[i*batch_size:(i+1)*batch_size]]

        i = (i + 1) % len(data)

        yield [imgs, embs, txts]


def get_data(filepath, split):
    print("Loading data")
    data_hdfs = h5py.File(filepath, 'r')[split]
    imgs = []
    embs = []
    txts = []
    for k in data_hdfs.keys():
        imgs.append(bytes(np.array(data_hdfs[k]['img'])))
        embs.append(np.array(data_hdfs[k]['embeddings']))
        txts.append(np.array(data_hdfs[k]['txt']))
    return [np.array(imgs), np.array(embs), np.array(txts)]


def log_text(texts, name, i, logger):
    texts = [str(t) for t in texts]
    texts = ' \n'.join(texts)
    logger.add_text('{}'.format(name, i), texts, i)


def log_images(images, name, i, logger):
    images = (images.reshape(4, 4, 64, 64, 3)
                    .transpose(0, 2, 1, 3, 4)
                    .reshape(4*64, 4*64, 3))
    images = ((images + 1) * 0.5)
    logger.add_image('{}'.format(name, i), images, i, dataformats='HWC')
