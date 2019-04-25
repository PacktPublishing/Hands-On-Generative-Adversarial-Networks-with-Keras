# spectral centroid
n_samples = 4
fig, axes = plt.subplots(2, n_samples, figsize=(6, 3))

# compute centroids on n_samples
for i in range(0, n_samples):
    # randomly sample mnist_train
    img = sample('mnist_train', 1)[0, 0]

    # compute the centroids
    centroids = (np.dot(np.arange(img.shape[0]).T, img) / 
                 (np.sum(img, axis=0) + np.finfo(float).eps))
    
    # plot the images and the centroids
    axes[0, i].imshow(img)
    axes[1, i].plot(centroids)

    # use a tight layout
    plt.tight_layout()


# spectral slope
n_samples = 4
fig, axes = plt.subplots(2, n_samples, figsize=(6, 3))

# instantiate linear model for computing the slope
lm = linear_model.LinearRegression()

# compute slope over 7 columns at a time
slope_len = 7

# compute slopes on n_samples
for i in range(0, n_samples):
    # randomly sample mnist_train
    img = sample('mnist_train', 1)[0, 0]

    # compute the centroids
    centroids = (np.dot(np.arange(img.shape[0]).T, img) / 
                 (np.sum(img, axis=0) + np.finfo(float).eps))
 
    # plot the images
    axes[0, i].imshow(img)
 
    # compute the slopes frame by frame with overlap
    X = np.arange(slope_len).reshape((-1, 1))
    slopes = np.array([lm.fit(X, centroids[i:i+slope_len]).coef_[0] 
                       for i in range(len(centroids) - slope_len + 1)])

    # plot the slopes
    axes[1, i].plot(slopes)

# use a tight layout
plt.tight_layout()


from scipy.stats import entropy
from numpy.linalg import norm
def JSD(P, Q):
    # Jensen-Shannon Divergence or Symmetric KL
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))


import matplotlib
%matplotlib inline
import matplotlib.pylab as plt
import pickle as pkl
import numpy as np
from scipy.stats import moment
import statsmodels.api as sm
from sklearn import linear_model
import gzip
from scipy.stats import ks_2samp


def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 1, 28, 28)
    return data / np.float32(256)


random_size = 10000
sample_size = 1000
labels = ('mnist_train', 'mnist_test', 'mnist_train_sigmoid', 'mnist_lsgan', 'mnist_iwgan',           
          'mnist_adversarial', 'random')
datasets = {}
datasets['mnist_test'] = load_mnist_images('t10k-images-idx3-ubyte.gz')
datasets['mnist_train'] = load_mnist_images('train-images-idx3-ubyte.gz'[
    :len(datasets['mnist_test'])]
datasets['mnist_adversarial'] = pkl.load(open('MNIST_test_adversarial_010.bin', "r"))
datasets['mnist_adversarial'] = datasets['mnist_adversarial'].reshape(-1, 1, 28, 28))
datasets['mnist_lsgan'] = np.load('lsgan_imgs.npy')[:len(datasets['mnist_test'])]
datasets['mnist_iwgan'] = np.load('iwgan_imgs.npy')[:len(datasets['mnist_test'])]

# use the statistics of the training data to generate binomial samples
datasets['random'] = np.random.binomial(1, datasets[labels[0]].mean(), (len(datasets['mnist_test']), 1, 28, 28))


# we are comparing distributions on other data to the distributions on the training data
ref = datasets[labels[0]].flatten() 

# compute the empirical histogram of the reference with 100 bins and within the [0, 1] range
ref_hist = np.histogram(ref, bins=100, range=(0.0, 1.0))[0]

fig, axes = plt.subplots(2, len(labels), figsize=(16, 4))
for i in range(len(labels)):
    tgt = datasets[labels[i]].flatten()
    
    # plot and compute the empirical histogram
    tgt_hist = axes[0, i].hist(tgt, bins=100, range=(0.0, 1.0))[0]
    
    # compute the ks 2 sample test and the jsd
    ts = ks_2samp(ref, tgt)
    jsd = JSD(ref_hist, tgt_hist)
    
    # plot the full distribution
    axes[0, i].set_title(labels[i])
 
    # plot and compute the empirical histogram on a new range
    tmin, tmax = 0.11, 0.88
    ids = (tgt > tmin) & (tgt < tmax)
    tgt_hist = axes[1, i].hist(tgt[ids], bins=100, range=(tmin, tmax))[0]
    print("{} {} JSD {}".format(labels[i], ts, jsd))
 
plt.tight_layout()


# flatten the reference distribution
ref = datasets[labels[0]].flatten()

# create a function to compute the empirical CDF of the reference distribution
ecdf_ref = sm.distributions.ECDF(ref)

# compute the empirical CD over 50 points evenly spaced between 0 and 1.
x = np.linspace(0, 1, num=50)
y_ref = ecdf_ref(x)

fig, axes = plt.subplots(1, len(labels), figsize=(16, 3))
axes = axes.flatten()

# iterate over the datasets
for i in range(len(labels)):
    # load the target dataset data
    tgt = datasets[labels[i]].flatten()

    # create a function to compute the empirical CDF of the target distribution
    ecdf_tgt = sm.distributions.ECDF(tgt)
    
    # compute and plot the empirical CDF of the target distribution
    y = ecdf_tgt(x)
    axes[i].set_ylim((0, 1.0))
    axes[i].step(x, y_ref, colors[1])
    axes[i].step(x, y, colors[2])
    axes[i].set_title(labels[i])

plt.suptitle('Empirical CDF')
plt.tight_layout()



