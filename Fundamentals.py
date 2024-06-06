import numpy as np
import os
import pandas as pd
import jax.numpy as jnp
from jax import grad, jacobian
from jax.scipy.special import gamma, gammaln


## Class that contains functions related to the dirichlet distribution that are necessary to optimize the hyperparameters \lambda

class Dirichlet:
    
    def __init__(self, alpha, J):
        self.alpha = alpha
        self.J = J
        
    
    ## In the following functions:
    
    ## sample_probs and sample_expert_probs are the same quantities but for multiple sets of covariates (J), each of which may have different partitions
    ## They are formatted as lists of arrays, with each array corresponding to one covariate set (thus J arrays in total)
    
    ## For both sample_probs and sample_expert_probs, the probabilities for each j = 1,...,J are in the j'th row
    
    
    ## Function to calculate the approximation of the MLE of alpha 
        
    def alpha_mle(self, total_model_probs, total_expert_probs, index = None):        
        
                
        ## If J=1, then we compute the MLE estimate of \alpha and return it    
        
        if self.J == 1:
            
            ## If J=1, then it is possible that the probabilities are encapsulated in a list, meaning that if we have e.g. prob = [0.5,0.5],
            ## the actual input is [[0.5, 0.5]]. In such a case, the parameter "index" will be 0 and the following two lines of code remove the outer list
            
            total_model_probs = total_model_probs[index] if index is not None else total_model_probs
            total_expert_probs = total_expert_probs[index] if index is not None else total_expert_probs
                                    
            assert jnp.isclose(jnp.sum(total_model_probs), 1) and jnp.isclose(jnp.sum(total_expert_probs), 1), "Probabilities must sum to 1"

            K = len(total_model_probs)
        
            kl_divergence = - jnp.sum(jnp.array([total_model_probs[k]*(jnp.log(total_expert_probs[k]) - jnp.log(total_model_probs[k])) for k in range(K)]))
                            
            return (K/2 - 1/2) / kl_divergence
        
        
        ## If J>1, we use a different formula
        
        assert jnp.all(jnp.isclose(jnp.array([jnp.sum(probs) for probs in total_model_probs]), jnp.ones(self.J))) and jnp.all(jnp.isclose(jnp.array([jnp.sum(probs) for probs in total_expert_probs]), jnp.ones(self.J))), "Probabilities must sum to 1"

        nom = 0
        den = 0
        
        for j in range(self.J):
            
            n_j = len(total_model_probs[j])
            
            nom += (n_j - 1)/2
            
            kl_divergence = - jnp.sum(jnp.array([total_model_probs[j][k]*(jnp.log(total_expert_probs[j][k]) - jnp.log(total_model_probs[j][k])) for k in range(n_j)]))
            
            den += kl_divergence
            
        return nom / den
    
    ## Simple function for the PDF of the Dirichet distribution (not used anywhere so far)
    
    def pdf(self, model_probs, expert_probs):
        
        assert jnp.isclose(jnp.sum(model_probs), 1) and jnp.isclose(jnp.sum(expert_probs), 1), "Probabilities must sum to 1"
        
        reset = 0
        
        if self.alpha is None:
            reset = 1
            self.alpha = self.alpha_mle(model_probs, expert_probs)
        
        num_1 = gamma(self.alpha)
        den_1 = np.prod([gamma(self.alpha*prob) for prob in model_probs])
        pt_1 = num_1 / den_1
                
        pt_2 = np.prod([expert_probs[i]**(self.alpha*model_probs[i] - 1) for i in range(len(model_probs))])
        
        if reset == 1: self.alpha = None
        
        return pt_1 * pt_2
    
    
    ## Function for log likelihood for J=1. We have as inputs sample_probs and sample_expert_probs and an index (j \in {1,...,J}).
    ## If we have a fixed \alpha as input, we use this as input for the computation, alternatively we compute it according to the
    ## MLE formula, using all the covariate sets (all j=1,...,J).
        
    def llik(self, total_model_probs, total_expert_probs, index=None):
        
        probs = total_model_probs[index] if index is not None else total_model_probs
        expert_probs = total_expert_probs[index] if index is not None else total_expert_probs
        
        assert jnp.all(jnp.isclose(jnp.array([jnp.sum(probs) for probs in total_model_probs]), jnp.ones(self.J))) and jnp.all(jnp.isclose(jnp.array([jnp.sum(probs) for probs in total_expert_probs]), jnp.ones(self.J))), "Probabilities must sum to 1"

        reset = 0
                
        if self.alpha is None:
            reset = 1
            self.alpha = self.alpha_mle(total_model_probs, total_expert_probs, index) ## we include all the probabilities to compute alpha!
                                
        loggamma_alpha = gammaln(self.alpha)
        
        num_1 = loggamma_alpha
        den_1 = np.sum(jnp.array([gammaln(self.alpha*probs)]))
        pt_1 = num_1 - den_1
        
        pt_2 = np.sum(jnp.array([(self.alpha*probs[i] - 1) * jnp.log(expert_probs[i]) for i in range(len(probs))]))
        
        if reset == 1: self.alpha = None
                                
        return pt_1 + pt_2
    
    
    ## Sum of log-likelihoods for j=1,...,J. Same as before, \alpha is either fixed or computed using the MLE formula
    
    def sum_llik(self, total_model_probs: list, total_expert_probs: list):
                
        if self.J == 1: return self.llik(total_model_probs, total_expert_probs, index = 0)
        
        assert jnp.all(jnp.isclose(jnp.array([jnp.sum(probs) for probs in total_model_probs]), jnp.ones(self.J))) and jnp.all(jnp.isclose(jnp.array([jnp.sum(probs) for probs in total_expert_probs]), jnp.ones(self.J))), "Probabilities must sum to 1"
        
        reset = 0
        
        if self.alpha is None:
            reset = 1
            self.alpha = self.alpha_mle(total_model_probs, total_expert_probs)
        
        total_llik = 0
        
        for j in range(self.J):
            
            total_llik += self.llik(total_model_probs, total_expert_probs, j)
            
        if reset == 1: self.alpha = None
            
        return total_llik
    
    ## Function to compute the gradient of the dirichlet log likelihood for one specific index (one j \in {1,...,J})
    ## with respect to this probability vector, using automatic differentiation. In order to compute this, we fix all other probabilitity vectors and
    ## we define the log likelihood with respect to the vector with respect to which we compute the gradient.
    ## This supports either fixed \alpha or using the MLE formula. In the latter case, the formula is dependent on the 
    ## vector we take the derivative with, meaning that we eventually take the derivative of the MLE formula.
    
    def grad_dirichlet_p(self, total_model_probs, total_expert_probs, index=None):
            
        def llik_index(sample_probs_index):
        
            # Replace the i-th probability vector in total_model_probs with total_model_probs[index], keeping the rest unchanged
            sample_probs_index_new = total_model_probs[:index] + [sample_probs_index] + total_model_probs[index+1:]
                        
            return self.llik(sample_probs_index_new, total_expert_probs, index)
        
        # Compute the gradient of llik_index with respect to total_model_probs
        return - grad(llik_index)(total_model_probs[index])
    



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
                
            J = samples.shape[1] if type(samples[1]) in [list, np.ndarray] else 1 ## Each column in "samples" corresponds to one set of covariates
            
            N_samples = samples.shape[0]
            
            N_classes = len(partitions)
            
            ## Here, the samples come from the prior predictive distribution and contain values for y, which is discrete
            
            ## In order to get the probabilities for each class c, we simply compute #(sample = c) / #(sample)
            
            model_probabilities = []
            
            
            for j in range(J):
                
                cov_set_j = samples[:,j] if J>1 else samples
                
                probs_list = np.zeros(N_classes)
                
                for i,C in enumerate(partitions):
                                        
                    probs_list[i] = np.sum(cov_set_j == C) / N_samples
                                    
                model_probabilities.append(probs_list)
                
                
        if self.target_type == "continuous":
                        
            
            J = samples.shape[1] if type(samples[1]) in [list, np.ndarray] else 1 ## Each column in "samples" corresponds to one set of covariates
    
            N_samples = samples.shape[0]
                
            ## We want the same format as the one of the elicited probabilities. For that reason,
            ## the output will be a list of probabilities
            
            model_probabilities = []
            
            for j in range(J):
                
                
                partition = np.copy(partitions[j]) if J>1 else np.copy(partitions)
                
                cov_set_j = samples[:,j] if J>1 else samples
                
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
    
    def __init__(self, alpha, J, ppd):
        super().__init__(alpha, J)
        self.ppd = ppd
        
    ### We assume that we have as input the prior probability distribution in closed form, for one partition.
    ### For instance, if Y~N(0,1) and we have a partition A=(a,b], then ppd = P(YεA) = Φ(b) - Φ(a)
    ### This input ("ppd") is assumed to have three parameters: 
    # 1) the partition (in the form of interval in the continuous case, or a single value in the discrete case).
    # 2) the hyperparameters "lam".
    # 3( the covariates of the model, if any. If there aren't any covariates, "ppd" is only defined by the first two inputs.
    
    ## To keep track of the dimensions, suppose that we have m hyperparameters and n partitions for a given covariate set j (j \in {1,...,J})
    
    ## We also account for the presence of covariate sets. Specifically, we add in some functions a parameter called "covariates" that is an array
    ## and represents each individual covariate set. The parameter "total_covariates" represents the quantity that contains all covariate sets and
    ## is either a list of covariate sets, or an array where each row corresponds to one covariate set. We initialize both parameters with None.
    ## If there are no covariates in the data, we simply have None in their place and they are not included in any computation.
    
    
    ## Here, we want to define the prior probability distribution for the partition of one covariate set j (j \in {1,...,J})).
    ## We will create an array that has as many components as there are partitions and contains in the i-th position the ppd for the i-th partition
    ## It takes as an input the partition, the hyperparameters \lambda and the covariates (if any).
    
    def ppd_function(self, partitions, lam, covariates=None):
                
        if covariates is not None:
            prior_pd = [self.ppd(partition, lam, covariates) for partition in partitions]
            
        else:
            prior_pd = [self.ppd(partition, lam) for partition in partitions]
        
        return jnp.array(prior_pd)  # shape (n, 1)
        
    ## Now, we compute the gradient (jacobian) of the prior probability distribution with respect to \lambda for one covariate set j (j \in {1,...,J})).
    
    def grad_ppd_lambda(self, partitions, lam, covariates=None):
        
        if covariates is not None:
            return jacobian(lambda lam: self.ppd_function(partitions, lam, covariates), argnums=0)(lam)  # shape (m, n)
        
        else:
            return jacobian(lambda lam: self.ppd_function(partitions, lam), argnums=0)(lam)  # shape (m, n)
    
    
    ## Finally, we compute the dirichlet likelihood gradient with respect to lambda. This will be used to perform gradient descent.
    
    def grad_dirichlet_lambda(self, lam, total_partitions, total_covariates, total_expert_probs, index):
                
        total_model_probs = [np.array(self.ppd_function(total_partitions[j], lam, total_covariates[j])) for j in range(self.J)]  ## The model probabilities given the hyperparameters \lambda for all j=1,...,J
        
        grad_dir_p = self.grad_dirichlet_p(total_model_probs, total_expert_probs, index)  ## The gradient of the Dirichlet llik with respect to the model probabilities for the j'th covariate set
        
        jac_p_lambda = self.grad_ppd_lambda(total_partitions[index], lam, total_covariates[index])
        
        dir_grad_lambda = jac_p_lambda.T@(grad_dir_p.T)
        
        dir_grad_lambda = dir_grad_lambda.T
        
        return dir_grad_lambda ## shape (m,1)
    
    ## Alternative computation of the dirichlet likelihood gradient with respect to \lambda. Here, we define the likelihood with
    ## respect to \lambda and we take the gradient right away, without the need of further computations and the use of chain rule. 
    
    def grad_dirichlet_lambda_2(self, lam, total_partitions, total_covariates, total_expert_probs, index):
        
        total_model_probs = lambda lam: [jnp.array(self.ppd_function(total_partitions[j], lam, total_covariates[j])) for j in range(self.J)]  ## The model probabilities given the hyperparameters \lambda for all j=1,...,J

        log_lik = lambda lam: self.llik(total_model_probs(lam), total_expert_probs, index)
        
        return -grad(log_lik, argnums=0)(lam) ## shape (m,1)
    
    
    ## If we have multiple covariate sets (J), we need to sum the gradients (implemented with "grad_dirichlet_lambda", although "grad_dirichlet_lambda_2") would lead to the exact same results.
    
    def sum_grad_dirichlet_lambda(self, total_partitions, lam, total_expert_probs, total_covariates=None):
                        
        total_dir_grad_lambda = np.zeros(len(lam))
        
        for j in range(self.J):
                        
            total_dir_grad_lambda += self.grad_dirichlet_lambda(lam, total_partitions, total_covariates, total_expert_probs, j)

        return total_dir_grad_lambda ## shape (m,1)
    
    
    def gradient_descent(self,
                         total_partitions,
                         total_expert_probs,
                         lam_0,
                         iters,
                         step_size,
                         tol,
                         total_covariates=None,
                         get_lik_and_grad_progression = True):
        
        lam_old = lam_0 ## initial value for the hyperparameters
                
        total_covariates = total_covariates if total_covariates is not None else [None]*self.J ## If we have covariates we leave them as is, otherwise we replace them with a list of None
        
        lik_progression = []
        grad_progression = []
        
        
        for i in range(iters):
            
            prev_model_probs = [np.array(self.ppd_function(total_partitions[j], lam_old, total_covariates[j])) for j in range(self.J)] 
                                
            prev_lik = self.sum_llik(prev_model_probs, total_expert_probs)
            
            lik_progression.append(prev_lik)
            
            grad_dir_lam = self.sum_grad_dirichlet_lambda(total_partitions, lam_old, total_expert_probs, total_covariates)
            
            grad_progression.append(np.linalg.norm(grad_dir_lam))
                        
            lam_new = lam_old - step_size * grad_dir_lam
                                
            curr_model_probs = [np.array(self.ppd_function(total_partitions[j], lam_new, total_covariates[j])) for j in range(self.J)]
                                                
            curr_lik = self.sum_llik(curr_model_probs, total_expert_probs)
            
            if abs(curr_lik - prev_lik) < tol: ## Stopping criterion: the dirichlet log likelihood changes less than "tol" between two iterations
                break
            
            lam_old = lam_new
        
        
        if get_lik_and_grad_progression:
            return lam_new, -jnp.array(lik_progression), jnp.array(grad_progression)
        
        return lam_new
    
    ## Function to get the \alpha estimate based on the MLE formula, using as inputs the expert probabilities and
    ## the partitions, hyperparameters lambda and covariate sets.
    
    def get_alpha(self, total_partitions, best_lam, total_expert_probs, total_covariates=None):
        
        total_covariates = total_covariates if total_covariates is not None else [None]*self.J
        
        best_model_probs = [np.array(self.ppd_function(total_partitions[j], best_lam, total_covariates[j])) for j in range(self.J)]
        
        index = 0 if self.J==1 else None
        
        alpha = self.alpha_mle(best_model_probs, total_expert_probs, index=index)
    
        return alpha
      