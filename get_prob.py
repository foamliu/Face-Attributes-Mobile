from scipy.stats import norm


def get_prob(beauty):
    mu = 49.1982
    sigma = 14.0220
    prob = norm.cdf(beauty, mu, sigma)
    return prob


if __name__ == '__main__':
    print(get_prob(80))
