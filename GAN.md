GAN

A framework for estimating generative models via an adversarial process, in which we simutanuously train two models: a generative model $G$ that captures the data distribution, and a discriminative model $D$ that estimates the probability that a sample came from the training data rather than $G$. The training procedure for $G$ is to maximize the probability of $D$ making a mistake, different from other generative models that is setted to recover the data distribution.

Deep generative models have difficulty of approximating many intractable probabilistic computations that arise in maximum likelihood estimation and related strategies.

$G$ generates samples by passing random noise, and can map the input distribution (Gaussian) to an arbitrary data distribution that we want to fit.

We train $G$ to minimize $\log(1-D(G(z)))$, but in the beginning of training, $G$ is not that good that this term is 0, making it hard to train. We can train G to maximize $\log D(G(z))$.

Algorithm:

Update D first, then G, $k$ is a hyperparameter to control D and G are comparable.

## Theoretical Results

Two parts: 1. this minimax game has a global optimum for $p_g = p_{\rm data}$. 2. Algorithm can optimize this equation and obtaining the desired result.
