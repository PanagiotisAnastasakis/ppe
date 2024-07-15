import numpy as np
import os
import pandas as pd

## that contains methods for processing inputs into probabilities in a form
## compatible with the optimization software used for PPE


class PPEProbabilities:
    
    '''
    Inputs:
    
    - "target_type" -> parameter for whether the target is continuous or discrete. Options: ["continuous", "discrete"]
    - "path" -> Boolean to denote whether we extract data from a path to a file or a folder for the expert probabilities
    
    '''

    def __init__(self, target_type, path):
        self.target_type = target_type
        self.path = path
        
        
    ## Function to process the expert's input (currently not used anywhere, and it assumes a very specific format for the input data)
    ## NOT RECOMMENDED TO USE
    ## This input is assumed to be either a path to a file/folder or a data matrix.
    

    def get_expert_data(self, expert_input):

        if self.target_type == "discrete":

            if self.path:
                if os.path.isfile(
                    expert_input
                ):  ## Checking if the input is a single file or a folder (which is assumed to contain files).
                    ## Note that if we have different number of partitions for different covariate sets, we need a folder to store them
                    expert_input = pd.read_csv(expert_input, index_col=0)
                    elicited_data = expert_input.to_numpy()

                else:  ## if not, then the path must lead to a folder containing multiple files.
                       ## In the discrete case, we assume that each file contains the classes in column 1 and the probabilities at column 2

                    files = os.listdir(expert_input)

                    # Filter only CSV files
                    csv_files = [file for file in files if file.endswith(".csv")]

                    elicited_covariate_sets = []

                    # Loop through each CSV file and process its contents
                    for csv_file in csv_files:
                        df = pd.read_csv(expert_input + "/" + csv_file, index_col=0)
                        elicited_covariate_set = df.to_numpy()

                        elicited_covariate_sets.append(elicited_covariate_set)

                    elicited_data = np.zeros(
                        (
                            elicited_covariate_sets[0].shape[0],
                            len(elicited_covariate_sets) + 1,
                        )
                    )

                    elicited_data[:, 0] = elicited_covariate_sets[0][:, 0]

                    for j, set in enumerate(elicited_covariate_sets):

                        elicited_data[:, j + 1] = set[:, -1]

            else:
                elicited_data = expert_input  ## the input is a matrix containing the classes and the corresponding probabilities

            elicited_data = elicited_data.astype(
                float
            )  ## ensuring that all values are numerical

            partitions = elicited_data[:, 0]

            expert_probabilities = [
                elicited_data[:, j + 1] for j in range(elicited_data.shape[1] - 1)
            ]

        if self.target_type == "continuous":

            ## Goal format: J separate matrices that have three columns; the first two being the partitions and third being the corresponding probabilities

            if self.path:
                if os.path.isfile(
                    expert_input
                ):  ## Checking if the input is a single file or a folder (which is assumed to contain files)
                    expert_input = pd.read_csv(expert_input, index_col=0)
                    elicited_data = expert_input.to_numpy()

                else:  ## if not, then the path must lead to a folder containing multiple files. In the continuous case, we assume that each file contains three columns; the first two being the partitions and third being the corresponding probabilities

                    files = os.listdir(expert_input)

                    # Filter only CSV files
                    csv_files = [file for file in files if file.endswith(".csv")]

                    elicited_data = []

                    # Loop through each CSV file and process its contents
                    for csv_file in csv_files:
                        df = pd.read_csv(expert_input + "/" + csv_file, index_col=0)
                        elicited_covariate_set = df.to_numpy()

                        elicited_data.append(elicited_covariate_set)

            else:
                elicited_data = expert_input  ## the input is a matrix containing the partitions and the corresponding probabilities

            elicited_data = [
                cov_set.astype(float) for cov_set in elicited_data
            ]  ## ensuring that all values are numerical

            partitions = [covariate_set[:, [0, 1]] for covariate_set in elicited_data]
            expert_probabilities = [
                covariate_set[:, -1] for covariate_set in elicited_data
            ]

        return partitions, expert_probabilities

        ## discrete data: "partitions" are an array containing the classes and "expert_probabilities" are a matrix with one column for each J
        ## continuous data: "partitions" are a list of length J, containing one partition for each covariate set and "expert_probabilities" is a list of same length, containing the respective probabilities

    ## Function for computing the prior predictive probabilities from samples of the prior predictive distribution
    
    '''
    Inputs:
    
    - "samples" -> Either a list of samples from the prior predictive distribution, or in the case that we elicit
                   probabilities for multiple covariate sets, a matrix where each column j corresponds to samples for
                   covariate set j, with jε{1,...,J}.
    - "partitions" -> A list of partitions for the target. If the target is discrete, then it has one element for each class.
                      If it is continuous, then it contains a list of intervals of the form [a,b] that cover all the target possible values
                      Same as with "samples", if we have multiple covariate sets, then it is a list of partitions for each j, thus
                      a list of length J.
    '''

    def ppd_probs(self, samples, partitions):

        if self.target_type == "discrete":

            J = (
                samples.shape[1] if type(samples[1]) in [list, np.ndarray] else 1
            )  ## Each column in "samples" corresponds to one set of covariates

            N_samples = samples.shape[0]

            N_classes = len(partitions)

            ## Here, the samples come from the prior predictive distribution and contain values for y, which is discrete

            ## In order to get the probabilities for each class c, we simply compute #(sample = c) / #(sample)

            model_probabilities = []

            for j in range(J):

                cov_set_j = samples[:, j] if J > 1 else samples

                probs_list = np.zeros(N_classes)

                for i, C in enumerate(partitions):

                    count = np.sum(cov_set_j == C)

                    if count == 0:
                        count = 1e-6 * N_samples

                    probs_list[i] = count / N_samples

                probs_list = probs_list / np.sum(probs_list)

                model_probabilities.append(probs_list)

        if self.target_type == "continuous":

            J = (
                samples.shape[1] if type(samples[1]) in [list, np.ndarray] else 1
            )  ## Each column in "samples" corresponds to one set of covariates

            N_samples = samples.shape[0]

            ## We want the same format as the one of the elicited probabilities. For that reason,
            ## the output will be a list of probabilities

            model_probabilities = []

            for j in range(J):

                partition = np.copy(partitions[j])

                cov_set_j = samples[:, j] if J > 1 else samples

                N_partitions = partition.shape[0]

                ## When sampling, it is possible that we get a value that is outside the partitions. In that case, we redifine the bounds according to the sampled value
                ## E.g. if the lower bound among all partitions is 15 and we sample the value 12, the new lower bound will be 12
                ## This however should not happen too often in the sampling process, as the lower and upper bounds should be wide enough to contain all samples

                sample_min = np.min(cov_set_j)
                sample_max = np.max(cov_set_j)

                if partition[0, 0] > sample_min:
                    partition[0, 0] = sample_min

                if partition[-1, 1] < sample_max:
                    partition[-1, 1] = sample_max

                probs_list = np.zeros(N_partitions)

                for i in range(N_partitions):

                    lower_bound = partition[i][0]
                    upper_bound = partition[i][1]

                    count = np.sum(
                        (cov_set_j >= lower_bound) & (cov_set_j <= upper_bound)
                    )

                    if count == 0:
                        count = 1e-6 * N_samples

                    probs_list[i] = count / N_samples

                probs_list = probs_list / np.sum(probs_list)

                model_probabilities.append(probs_list)

        return model_probabilities
