import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import numpy as np
from utils import get_data, iterate_minibatches, load_model


def infer(data_filepath='data/flowers.hdf5', z_dim=128, out_dir='gan',
          n_samples=5):

    G = load_model(out_dir)
    val_data = get_data(data_filepath, 'train')
    val_data = next(iterate_minibatches(val_data, n_samples))    
    emb, txts = val_data[1], val_data[2]

    # sample z vector for inference
    z = np.random.uniform(-1, 1, size=(n_samples, z_dim))

    G.trainable = False
    fake_images = G.predict([z, emb])
    for i in range(n_samples):
        img = ((fake_images[i] + 1)*0.5)
        plt.imsave("{}/fake_{}".format(out_dir, i), img)
        print(i, str(txts[i]).strip(),
              file=open("{}/fake_text.txt".format(out_dir), "a"))


infer()
