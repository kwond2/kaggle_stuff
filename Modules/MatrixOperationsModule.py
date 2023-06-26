import jax.numpy as jnp

## Helper class that references the jnp array and has helpful operations attached automatically
class MatrixOps:
    def __init__(self, shape=None, A=None):
        self.n, self.d = shape #expects np shape
        self.A = A # matrix or vector

    @classmethod
    def meanVectorOTF(cls, A):
        n = A.shape[0]
        return ((jnp.ones(n).T@A)/n).T
    
    ## Generic On The Fly Mean Centering. Expects an n x d numpy array representing a matrix A.
    @classmethod
    def meanCenterOTF(cls, A):
        n = A.shape[0]
        mu = meanVectorOTF(A)
        return A - jnp.outer(mu, jnp.ones(d))

    
    ## induced matrix norm computations
    @classmethod
    def inducedNorm(cls, args):
        ## TODOs
        return None