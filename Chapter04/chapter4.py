# get the training data D, sample the Generator with random z to produce r
N = X_train
z = np.random.uniform(-1, 1, (1, z_dim))
r = G.predict_on_batch(z)

# define our distance measure S to be L1
S = lambda n, r: np.sum(np.abs(n - r))

# compute the distances between the reference and the samples in N using the measure D
distances = [D(n, r) for n in N]

# find the indices of the most similar samples and select them from N
nearest_neighbors_index = np.argpartition(distances, k)
nearest_neighbors_images = N[nearest_neighbors_index]

# generate fake images from the discriminator
n_fake_images = 5000
z = np.random.uniform(-1, 1, (n_fake_images, z_dim))
x = G.predict_on_batch(z)

def compute_inception_score(x, inception_model, n_fake_images, z_dim):
    # probability of y given x
    p_y_given_x = inception_model.predict_on_batch(x)

# marginal probability of y
 q_y = np.mean(p_y_given_x, axis=0)

inception_scores = p_y_given_x * (np.log(p_y_given_x) - np.log(q_y)
    inception_score = np.exp(np.mean(inception_scores))
    return inception_score


def get_mean_and_covariance(data):
     mean = np.mean(data, axis=0)
     covariance = np.cov(data, rowvar=False) # rowvar?
     return mean, covariance

def compute_frechet_inception_distance(mean_r, mean_f, cov_r, cov_f):
    l2_mean = np.sum((mean_r - mean_f)**2)
    cov_mean, _ = np.trace(scipy.linalg.sqrtm(np.dot(cov_r, cov_f)))
    return l2_mu + np.trace(cov_r) + np.trace(cov_f) - 2 * np.trace(cov_mean)

