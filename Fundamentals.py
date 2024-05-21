from scipy.special import gamma, loggamma, digamma
import numpy as np
import os
import pandas as pd
import jax.numpy as jnp
from jax import jacobian



class Dirichlet:
    
    def __init__(self, alpha):
        self.alpha = alpha
        
    ## just a test    
        
    ## In the following functions:
    
    ## probs correspond to the prior predictive distribution probabilities
    ## expert_probs correspond to the elicited probabilities from the expert
    
    ## sample_probs and sample_expert_probs are the same quantities but for multiple sets of covariates (J), each of which may have different partitions
    
    ## For both sample_probs and sample_expert_probs, the probabilities for each j = 1,...,J are in the j'th row
    
    
    ## Function to calculate the approximation of the MLE of alpha for J=1
        
    def alpha_mle(self, probs, expert_probs):
        
        #assert probs.ndim == 1 and expert_probs.ndim == 1, "This operation requires one set of probabilities only"
        assert np.isclose(np.sum(probs), 1) and np.isclose(np.sum(expert_probs), 1), "Probabilities must sum to 1"
        
        K = len(probs)
        
        kl_divergence = - np.sum([probs[k]*(np.log(expert_probs[k]) - np.log(probs[k])) for k in range(K)])
        
        return (K/2 - 1/2) / kl_divergence
    
    
    ## Function to calculate the same quantity for J>1
        
    def alpha_mle_multiple_samples(self, sample_probs, sample_expert_probs):
        
        J = len(sample_probs) if type(sample_probs[0]) in [list, np.ndarray] else 1
                
        if J == 1: return self.alpha_mle(sample_probs, sample_expert_probs)
                
        assert np.all(np.isclose(np.array([np.sum(probs) for probs in sample_probs]), np.ones(J))) and np.all(np.isclose(np.array([np.sum(probs) for probs in sample_expert_probs]), np.ones(J))), "Probabilities must sum to 1"
        
        nom = 0
        den = 0
        
        for j in range(J):
            
            n_j = len(sample_probs[j])
            
            nom += (n_j - 1)/2
            
            kl_divergence = - np.sum([sample_probs[j][k]*(np.log(sample_expert_probs[j][k]) - np.log(sample_probs[j][k])) for k in range(n_j)])
            
            den += kl_divergence
            
        return nom / den
    
    ## Simple function for the PDF of the Dirichet distribution
    
    def pdf(self, probs, expert_probs):
        
        #assert probs.ndim == 1 and expert_probs.ndim == 1, "Pdf is defined for one set of probabilities only"
        assert np.isclose(np.sum(probs), 1) and np.isclose(np.sum(expert_probs), 1), "Probabilities must sum to 1"
        
        reset = 0
        
        if self.alpha is None:
            reset = 1
            self.alpha = self.alpha_mle_multiple_samples(probs, expert_probs)
        
        num_1 = gamma(self.alpha)
        den_1 = np.prod([gamma(self.alpha*prob) for prob in probs])
        pt_1 = num_1 / den_1
                
        pt_2 = np.prod([expert_probs[i]**(self.alpha*probs[i] - 1) for i in range(len(probs))])
        
        if reset == 1: self.alpha = None
        
        return pt_1 * pt_2
    
    
    ## Function for log likelihood for J=1
        
    def llik(self, probs, expert_probs):
                
        #assert probs.ndim == 1 and expert_probs.ndim == 1, "Likelihood is defined for one set of probabilities only"
        
        assert np.isclose(np.sum(probs), 1) and np.isclose(np.sum(expert_probs), 1), "Probabilities must sum to 1"
        
        reset = 0
        
        if self.alpha is None:
            reset = 1
            self.alpha = self.alpha_mle_multiple_samples(probs, expert_probs)
        
        loggamma_alpha = loggamma(self.alpha)
        
        num_1 = loggamma_alpha
        den_1 = np.sum([loggamma_alpha + loggamma(prob) for prob in probs])
        pt_1 = num_1 - den_1
        
        pt_2 = np.sum([(self.alpha*probs[i] - 1) * np.log(expert_probs[i]) for i in range(len(probs))])
        
        if reset == 1: self.alpha = None
        
        return pt_1 + pt_2
    
    ## Sum of log-likelihoods. This will be used in later stages during optimization
    
    def sum_llik(self, sample_probs: list, sample_expert_probs: list):
        
        J = len(sample_probs) if type(sample_probs[0]) in [list, np.ndarray] else 1
                
        if J == 1: return self.llik(sample_probs, sample_expert_probs)
        
        assert np.all(np.isclose(np.array([np.sum(probs) for probs in sample_probs]), np.ones(J))) and np.all(np.isclose(np.array([np.sum(probs) for probs in sample_expert_probs]), np.ones(J))), "Probabilities must sum to 1"
        
        reset = 0
        
        if self.alpha is None:
            reset = 1
            self.alpha = self.alpha_mle_multiple_samples(sample_probs, sample_expert_probs)
        
        total_llik = 0
        
        for j in range(J):
            
            total_llik += self.llik(sample_probs[j], sample_expert_probs[j])
            
        if reset == 1: self.alpha = None
            
        return total_llik
    
    
    ### Function to compute the derivative of the dirichlet negtive log likelihood for one partition P_λ. It assumes a constant value for alpha.
    ### Formula implemented according to T. P. Minka. Estimating a Dirichlet distribution, 2000.
    
    def grad_dirichlet_p(self, probs, expert_probs):
    
        sum_val = np.sum(probs*(np.log(expert_probs) - digamma(self.alpha*probs)))
        
        dlogD = np.zeros(len(probs))
        
        for i in range(len(probs)):
        
            dlogD[i] = len(probs)*self.alpha*(np.log(expert_probs[i]) - digamma(self.alpha*probs[i]) - sum_val)
            
        return -np.array(dlogD)
    



class PPEProbabilities:
    
    def __init__(self, target_type, path):
        self.target_type = target_type
        self.path = path
        
    def get_expert_data(self, expert_input):        
        
        if self.target_type == "discrete":
            
            if self.path:
                if os.path.isfile(expert_input):  ## Checking if the input is a single file or a folder (which is assumed to contain files). Note that if we have different number of partitions for different covariate sets, we need a folder to store them
                    expert_input = pd.read_csv(expert_input, index_col=0)
                    elicited_data = expert_input.to_numpy()
                     
                else: ## if not, then the path must lead to a folder containing multiple files. In the discrete case, we assume that each file contains the classes in column 1 and the probabilities at column 2
                    
                    files = os.listdir(expert_input)

                    # Filter only CSV files
                    csv_files = [file for file in files if file.endswith('.csv')]
                    
                    elicited_covariate_sets = []

                    # Loop through each CSV file and process its contents
                    for csv_file in csv_files:
                        df = pd.read_csv(expert_input + "/" + csv_file, index_col=0)
                        elicited_covariate_set = df.to_numpy()
                        
                        elicited_covariate_sets.append(elicited_covariate_set)
                        
                    elicited_data = np.zeros((elicited_covariate_sets[0].shape[0], len(elicited_covariate_sets) + 1))
                    
                    elicited_data[:,0] = elicited_covariate_sets[0][:,0]
                    
                    for j, set in enumerate(elicited_covariate_sets):
                        
                        elicited_data[:,j+1] = set[:,-1]
                        
            else:
                elicited_data = expert_input   ## the input is a matrix containing the classes and the corresponding probabilities
                
        
            elicited_data = elicited_data.astype(float)  ## ensuring that all values are numerical      
            
            partitions = elicited_data[:,0]
            
            expert_probabilities = [elicited_data[:,j+1] for j in range(elicited_data.shape[1] - 1)]
        
        if self.target_type == "continuous":
            
            ## Goal format: J separate matrices that have three columns; the first two being the partitions and third being the corresponding probabilities
            
            if self.path:
                if os.path.isfile(expert_input):  ## Checking if the input is a single file or a folder (which is assumed to contain files)
                    expert_input = pd.read_csv(expert_input, index_col=0)
                    elicited_data = expert_input.to_numpy()
                     
                else: ## if not, then the path must lead to a folder containing multiple files. In the continuous case, we assume that each file contains three columns; the first two being the partitions and third being the corresponding probabilities
                    
                    files = os.listdir(expert_input)

                    # Filter only CSV files
                    csv_files = [file for file in files if file.endswith('.csv')]
                    
                    elicited_data = []

                    # Loop through each CSV file and process its contents
                    for csv_file in csv_files:
                        df = pd.read_csv(expert_input + "/" + csv_file, index_col=0)
                        elicited_covariate_set = df.to_numpy()
                        
                        elicited_data.append(elicited_covariate_set)
                        
                    
                        
            else:
                elicited_data = expert_input   ## the input is a matrix containing the partitions and the corresponding probabilities
                
        
            elicited_data = [cov_set.astype(float) for cov_set in elicited_data]  ## ensuring that all values are numerical
            
            partitions = [covariate_set[:,[0,1]] for covariate_set in elicited_data]
            expert_probabilities = [covariate_set[:, -1] for covariate_set in elicited_data]
            
            
        return partitions, expert_probabilities
    
    ## discrete data: "partitions" are an array containing the classes and "expert_probabilities" are a matrix with one column for each J
    ## continuous data: "partitions" are a list of length J, containing one partition for each covariate set and "expert_probabilities" is a list of same length, containing the respective probabilities
    
    
    
    def ppd_probs(self, samples, partitions):
                
        
        if self.target_type == "discrete":
                
            J = samples.shape[1] ## Each column in "samples" corresponds to one set of covariates
            
            N_samples = samples.shape[0]
            
            N_classes = len(partitions)
            
            ## Here, the samples come from the prior predictive distribution and contain values for y, which is discrete
            
            ## In order to get the probabilities for each class c, we simply compute #(sample = c) / #(sample)
            
            model_probabilities = []
            
            
            for j in range(J):
                
                cov_set_j = samples[:,j]
                
                probs_list = np.zeros(N_classes)
                
                for i,C in enumerate(partitions):
                                        
                    probs_list[i] = np.sum(cov_set_j == C) / N_samples
                                    
                model_probabilities.append(probs_list)
                
                
        if self.target_type == "continuous":
                        
            
            J = samples.shape[1] ## Each column in "samples" corresponds to one set of covariates
    
            N_samples = samples.shape[0]
                
            ## We want the same format as the one of the elicited probabilities. For that reason,
            ## the output will be a list of probabilities
            
            model_probabilities = []
            
            for j in range(J):
                
                
                partition = np.copy(partitions[j])
                cov_set_j = samples[:,j]
                
                N_partitions = partition.shape[0]
                
                ## When sampling, it is possible that we get a value that is outside the partitions. In that case, we redifine the bounds according to the sampled value
                ## E.g. if the lower bound among all partitions is 15 and we sample the value 12, the new lower bound will be 12
                ## This however should not happen too often in the sampling process, as the lower and upper bounds should be wide enough to contain all samples

                sample_min = np.min(cov_set_j)
                sample_max = np.max(cov_set_j)
                
                if partition[0,0] > sample_min:
                    partition[0,0] = sample_min
                    
                if partition[-1,1] < sample_max:
                    partition[-1,1] = sample_max
                    
                probs_list = np.zeros(N_partitions)
                    
                for i in range(N_partitions):
                    
                    lower_bound = partition[i][0]
                    upper_bound = partition[i][1]
                                        
                    count = np.sum((cov_set_j >= lower_bound) & (cov_set_j <= upper_bound))
                    
                    probs_list[i] = count / N_samples
                                            
                model_probabilities.append(probs_list)


        return model_probabilities
    
    
    
class optimize_ppe(Dirichlet): ### closed form is assumed!!!
    
    def __init__(self, alpha, ppd):
        super().__init__(alpha)
        self.ppd = ppd
        
    ### We assume that we have as input the prior probability distribution in closed form, for one partition.
    ### For instance, if Y~N(0,1) and we have a partition A=(a,b], then ppd = P(YεA) = Φ(b) - Φ(a)
    
    ### The inputs of ppd are 
    # 1) the partition (in the form of interval in the continuous case, or a single value in the discrete case)
    # 2) the hyperparameters lam.
    
    ## To keep track of the dimensions, suppose that we have m hyperparameters and n partitions
    
    ## We also account for the presence of covariate sets. Specifically, we add in some functions a parameter called "covariates" that is a np.array
    ## and represents each individual covariate set. The parameter "total_covariates" represents the quantity that contains all covariate sets and
    ## is either a list of covariate sets, or a np.array where each row corresponds to one covariate set. We initialize both parameters with None.
    
    ## Here, we want to define the prior probability distribution for all partitions
    ## We will create an array that has as many components as there are partitions and contains in the i-th position the ppd for the i-th partition
    
    def ppd_function(self, partitions, lam, covariates=None):
                
        if covariates is not None:
            prior_pd = [self.ppd(partition, lam, covariates) for partition in partitions]
            
        else:
            prior_pd = [self.ppd(partition, lam) for partition in partitions]
        
        return jnp.array(prior_pd)  # shape (n, 1)
        
    ## Now, we compute the gradient (jacobian) of the prior probability distribution with respect to lambda
    
    def grad_ppd_lambda(self, partitions, lam, covariates=None):
        
        if covariates is not None:
            return jacobian(lambda lam: self.ppd_function(partitions, lam, covariates), argnums=0)(lam)  # shape (m, n)
        
        else:
            return jacobian(lambda lam: self.ppd_function(partitions, lam), argnums=0)(lam)  # shape (m, n)
    
    
    ## Finally, we compute the dirichlet likelihood gradient with respect to lambda. This will be used to perform gradient descent
    
    def grad_dirichlet_lambda(self, partitions, lam, expert_probs, covariates=None):
        
        model_probs = self.ppd_function(partitions, lam, covariates)
        
        grad_dir_p = self.grad_dirichlet_p(model_probs, expert_probs)
                
        dir_grad = self.grad_ppd_lambda(partitions, lam, covariates).T@(grad_dir_p.T)
        
        dir_grad = dir_grad.T
        
        return dir_grad ## shape (m,1)
    
    
    ## If we have multiple covariate sets (J), we need to sum the gradients
    
    def sum_grad_dirichlet_lambda(self, total_partitions, lam, total_expert_probs, total_covariates=None):
        
        ## We assume that total_partitions is a list of partitions
        
        total_dir_grad = np.zeros(len(lam))
        
        for j in range(len(total_partitions)):
            
            covariates = total_covariates[j] if total_covariates is not None else None

            total_dir_grad += self.grad_dirichlet_lambda(total_partitions[j], lam, total_expert_probs[j], covariates)

        return total_dir_grad ## shape (m,1)
    
    ## Performing gradient descent to optimize the hyperparameters
    
    def gradient_descent(self, total_partitions, total_expert_probs, lam_0, iters, step_size, tol, total_covariates=None, get_lik_progression = True):
        
        lam_old = lam_0
        
        lik_progression = []

        
        for i in range(iters):
            
            prev_model_probs = [np.array(self.ppd_function(partitions, lam_old, total_covariates[j] if total_covariates is not None else None)) for j, partitions in enumerate(total_partitions)]

                                    
            prev_lik = self.sum_llik(prev_model_probs, total_expert_probs)
            
            lik_progression.append(prev_lik)
                        
            lam_new = lam_old - step_size * self.sum_grad_dirichlet_lambda(total_partitions, lam_old, total_expert_probs, total_covariates)
                        
            curr_model_probs = [np.array(self.ppd_function(partitions, lam_new, total_covariates[j] if total_covariates is not None else None)) for j, partitions in enumerate(total_partitions)]

            curr_lik = self.sum_llik(curr_model_probs, total_expert_probs)
            
            if abs(curr_lik - prev_lik) < tol:
                break
            
            lam_old = lam_new
        
        
        if get_lik_progression:
            return lam_new, -np.array(lik_progression)
        
        return lam_new
    
    ## Function to get the concentration parameter alpha, indicating how good of a fit this is
    
    def get_alpha(self, total_partitions, best_lam, total_expert_probs, total_covariates=None):
        
        best_model_probs = [np.array(self.ppd_function(partitions, best_lam, total_covariates[j] if total_covariates is not None else None)) for j, partitions in enumerate(total_partitions)]
        
        alpha = self.alpha_mle_multiple_samples(best_model_probs, total_expert_probs)
    
        return alpha