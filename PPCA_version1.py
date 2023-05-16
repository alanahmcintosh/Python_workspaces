import pandas as pd # pandas to convert my csv to a dataframe
import numpy as np 
import numpy.random as npr
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import seaborn as sns
import random
from numpy import arange
import scipy.sparse
from scipy.sparse import coo_matrix
from scipy import sparse, stats
from scipy import linalg



class ppca():
    def __init__(self, factors = 2, sigma2 = None):
        self.factors = factors
        self.ran = False
        self.sigma2 = sigma2
        #self.q=factors
    def generate(self, n, standardise = True):
        if self.ran:
            gen_x = np.zeros((n*self.n, self.D))
            z = np.zeros((n*self.n, self.factors))
            jf = 0
            for i in range(n):
                for j in range(self.n):
                    z[j+jf,:] = np.random.multivariate_normal(self.z_mu[:,j+jf], self.z_cov)
                jf += self.n
            gen_x = np.matmul(z,np.transpose(self.W)) + self.means
            return(gen_x, z)
        else:
            print ("Use max_likelihood() to learn parameters first. ")


    def max_likelihood(self, x, standardise = True, mask = None):
        self.n  = x.shape[0]
        self.D = x.shape[1]
        self.x = x + 0.
        # standardise x by subtracting the mean and dividing by the standard deviation
        if standardise:
            means = np.zeros((x.shape[1],))
            for i in range(x.shape[1]):
                means[i] = np.mean(x[:,i])
                self.x[:,i] = (x[:,i] - means[i])/(np.var(x[:,i])**.5) 
        else:
            # keep means and delete from x. 
            means = np.zeros((x.shape[1],))
            for i in range(x.shape[1]):
                means[i] = np.mean(x[:,i])
                self.x[:,i] = x[:,i] - means[i]
        # Calculate S, the covariance matrix of the observed data
        if mask is None:
            S = 1/self.n * np.matmul(np.transpose(self.x), self.x)
        else:
            mask = np.array(mask)  # ensure mask is a numpy array
            if mask.shape != x.shape:
                raise ValueError("Mask shape must match data shape.")
            # apply the mask
            masked_x = np.where(mask, x, 0)
            # calculate S using the masked data
            S = 1/self.n * np.matmul(np.transpose(masked_x), masked_x)
            # restore the original x array
            self.x = x

        # do eigenvalue decomposition
        L, U = np.linalg.eigh(S)
        # sort eigenvalues in decreasing order
        idx_l = np.argsort(-L)
        U = U[:,idx_l]
        L = L[idx_l]
        if self.sigma2 == None:
            # get sigma2 maximums likelihood estimation. 
            self.sigma2 = 1/(self.D - self.factors)*np.sum(L[self.factors:]) 
        # get W using the first factors eignevectors of S
        U = U[:,0:self.factors]
        L_diag = np.diag(L[0:self.factors])
        self.W = np.matmul(U, (L_diag - self.sigma2*np.eye(self.factors))**.5)
        # calculate M, an axuiliary variable useful for the rest of calculations.
        self.M  = np.matmul(np.transpose(self.W), self.W) + self.sigma2*np.eye(self.factors)
        try:
            self.M_inv = np.linalg.pinv(self.M, rcond=1e-3) # increase rcond value to improve convergence
        except:
            print('SVD did not converge')
            return
        # calcualte the covariance matrix of the joint distributions of x and z. 
        self.C = np.matmul(self.W, np.transpose(self.W)) + self.sigma2*np.eye(self.D) # cpvariance matrix of the joing distribution s of x and z
        self.C_inv = self.sigma2**-1*np.eye(self.D) - self.sigma2*np.matmul(np.matmul(self.W, self.M_inv), np.transpose(self.W)) # inverse of C
        self.z_mu = np.matmul(np.matmul(self.M_inv, np.transpose(self.W)), np.transpose(self.x)) # the mean of the latenet variables z
        self.z_cov = self.sigma2*self.M # covarianc e matrix of x
        self.ran = True
        self.U = U # array containing the eigenvextors of the covariance matrix S sorted in decreasing order of eigenvalues
        self.L = L_diag #
        self.means = means
        self.S = S # the covariance matrix of the obs data
    def get_W_cov(self):
        M_inv = self.M_inv
        sigma2 = self.sigma2
        W_cov = sigma2 * M_inv
        return W_cov
    
    def predictive_check(x, factors, holdout_portion=0.2, n_rep=100):
        factors = factors
        n, D = x.shape
        n_holdout = int(holdout_portion * n * D)

        # censor at random some parts of x.
        holdout_row = np.random.randint(n, size=n_holdout)
        holdout_col = np.random.randint(D, size=n_holdout)
        holdout_mask = (sparse.coo_matrix((np.ones(n_holdout),  (holdout_row, holdout_col)), shape = x.shape)).toarray()
        holdout_subjects = np.unique(holdout_row)
        # # use holdout_mask to generate training and validation set. 
        x_train  = np.multiply(1-holdout_mask, x)
        x_train = x_train[np.logical_not(np.isnan(x_train))]
        a, b = x_train.shape
        x_val = np.multiply(holdout_mask, x)
        x_train = x_train.reshape(a, b)
        holdout_gen = np.zeros((n_rep,*(x_train.shape)))
        # # learn the parameters and learn the probabilistic ppca.
        m_ppca = ppca(factors)   
        m_ppca.max_likelihood(x_train, standardise = False)
        gen_x, unk = m_ppca.generate(1) #!!!! VERY IMPORTANT, generate by default returns two arrays, must remember to give it to things
        for i in range(factors):
            holdout_gen[i] = np.multiply(gen_x, holdout_mask)
        
        w_mean = np.mean(m_ppca.W.flatten())
        z_mean=(np.mean(m_ppca.z_mu.flatten()))
        W_cov = m_ppca.get_W_cov()
        W_std = np.sqrt(np.diag(W_cov))
        z_covariance = m_ppca.z_cov
        z_standard_deviation = np.sqrt(np.diag(z_covariance))
        n_eval = 100 # we draw samples from the inferred Z and W
        obs_ll = []
        rep_ll = []
        for j in range(n_eval):
            w_sample = npr.normal(w_mean, W_std)
            z_sample = npr.normal(z_mean, z_standard_deviation)
            holdoutmean_sample = np.multiply(z_sample.dot(w_sample), holdout_mask)
            obs_ll.append(np.mean(stats.norm(holdoutmean_sample, 0.1).logpdf(x_val), axis=1))
            rep_ll.append(np.mean(stats.norm(holdoutmean_sample, 0.1).logpdf(holdout_gen),axis=2))
        obs_ll_per_zi, rep_ll_per_zi = np.mean(np.array(obs_ll), axis=0), np.mean(np.array(rep_ll), axis=0)
        num_datapoints, data_dim = x_train.shape
        pvals = np.array([np.mean(rep_ll_per_zi[:,i] < obs_ll_per_zi[i]) for i in range(num_datapoints)])
        holdout_subjects = np.unique(holdout_row)
        overall_pval = np.mean(pvals[holdout_subjects])
        print("Predictive check p-values", overall_pval)