import numpy as np

#function will take the previously observed data
def monte_carlo_sim(D_prev , mu_t, tau_t, alpha_t, beta_t, P_initial, iterations):
    tau_proposed = P_initial
    n = len(D_prev)
    d_values = []
    for i in range(iterations):
        ## This will calculate the hypermarameter mu of unknown posterior parameter Mu.
        mu_posterior = (tau_t * mu_t + tau_proposed * np.sum(D_prev)) / (tau_t + n * tau_proposed)

        ## this will calculate the hyperparameter tau of the unknown posterior parameter mu.
        tau_posterior = (tau_t + n * tau_proposed)

        ## Now calculate a random value using the Gaussian Distribution defined by the parameters Mu_posterior and Tau_post.
        ran_mu = np.random.normal(mu_posterior , 1/np.sqrt(tau_posterior))
        #The above calculated value is our proposed value for the first trial
        mu_prop = ran_mu
        ## Now using the proposed  value of Mu (mu_prop) to find the hyperparameters of unknown parameter tau.

        alpha = alpha_t + n/2 # value of hyperparameter alpha

        beta = beta_t + np.sum((D_prev - mu_prop)**2)/2
        ## Now draw a value from the Gamma distribution uing above calculated value

        ##### IMPORTANT ######
        ## The Gamma function in the numpy library uses Scale parameter and not the rate (Beta) that we calculated
        ''' Remember that we are calculating the rate parameter for the gamma distribution but
            Numpy will accept Scale parameter so remember to convert it first '''
        ran_tau = np.random.gamma(alpha , 1/beta)

        tau_proposed = ran_tau
        d_values.append(mu_prop)

    return d_values
