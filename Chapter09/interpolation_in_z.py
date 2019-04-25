import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import numpy as np
from data_utils import get_data, iterate_minibatches, load_model


def infer(data_filepath='data/flowers.hdf5', z_dim=128, out_dir='gan',
          n_steps=10):

    G = load_model(out_dir)
    val_data = get_data(data_filepath, 'train')
    val_data = next(iterate_minibatches(val_data, 1))
    emb_fixed, txt_fixed = val_data[1], val_data[2]
    
    z_start = np.random.uniform(-1, 1, size=(1, z_dim))
    z_end = np.random.uniform(-1, 1, size=(1, z_dim))

    G.trainable = False
    for i in range(n_steps+1):
        p = i/float(n_steps)
        z = z_start * (1-p) + z_end * p
        fake_image = G.predict([z, emb_fixed])[0]
        img = ((fake_image + 1)*0.5)
        plt.imsave("{}/fake_z_interpolation_i{}".format(out_dir, i), img)
        print(i, str(txt_fixed[0]).strip(),
              file=open("{}/fake_z_interpolation.txt".format(out_dir), "a"))

infer()
