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
    emb_a, emb_b = val_data[1]
    txts = val_data[2]

    # add batch dimension
    emb_a, emb_b = emb_a[None, :], emb_b[None, :]

    # sample z vector for inference
    z = np.random.uniform(-1, 1, size=(1, z_dim))

    G.trainable = False
    # predict using embeddings a and b
    fake_image_a = G.predict([z, emb_a])[0]
    fake_image_b  = G.predict([z, emb_b])[0]

    # add and subtract
    emb_add = (emb_a + emb_b)
    emb_a_sub_b = (emb_a - emb_b)
    emb_b_sub_a = (emb_b - emb_a)

    # generate images
    fake_a = G.predict([z, emb_a])[0]
    fake_b = G.predict([z, emb_b])[0]
    fake_add = G.predict([z, emb_add])[0]
    fake_a_sub_b = G.predict([z, emb_a_sub_b])[0]
    fake_b_sub_a = G.predict([z, emb_b_sub_a])[0]

    fake_a = ((fake_a + 1)*0.5)
    fake_b = ((fake_b + 1)*0.5)
    fake_add = ((fake_add + 1)*0.5)
    fake_a_sub_b = ((fake_a_sub_b + 1)*0.5)
    fake_b_sub_a = ((fake_b_sub_a + 1)*0.5)

    plt.imsave("{}/fake_text_arithmetic_a".format(out_dir), fake_a)
    plt.imsave("{}/fake_text_arithmetic_b".format(out_dir), fake_b)
    plt.imsave("{}/fake_text_arithmetic_add".format(out_dir), fake_add)
    plt.imsave("{}/fake_text_arithmetic_a_sub_b".format(out_dir), fake_a_sub_b)
    plt.imsave("{}/fake_text_arithmetic_b_sub_a".format(out_dir), fake_b_sub_a)
    print(str(txts[0]), str(txts[1]),
          file=open("{}/fake_text_arithmetic.txt".format(out_dir), "a"))


infer()
