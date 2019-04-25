import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import numpy as np
from utils import get_data, iterate_minibatches, load_model


def infer(data_filepath='data/flowers.hdf5', z_dim=128, out_dir='gan',
          n_steps=10):

    G = load_model(out_dir)
    val_data = get_data(data_filepath, 'train')
    val_data = next(iterate_minibatches(val_data, 2))
    emb_source, emb_target = val_data[1]
    txts = val_data[2]

    z = np.random.uniform(-1, 1, size=(1, z_dim))

    G.trainable = False
    for i in range(n_steps+1):
        p = i/float(n_steps)
        emb = emb_source * (1-p) + emb_target * p
        emb = emb[None, :]
        fake_image = G.predict([z, emb])[0]
        img = ((fake_image + 1)*0.5)
        plt.imsave("{}/fake_text_interpolation_i{}".format(out_dir, i), img)
        print(i, str(txts[int(round(p))]).strip(),
              file=open("{}/fake_text_interpolation.txt".format(out_dir), "a"))


infer()
