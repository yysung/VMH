import csv
import pandas as pd
import numpy as np
import json
from collections import defaultdict
import matplotlib.pyplot as plt

def pre_mace(final_dff):
    uni_workers = np.unique(final_dff['workerid'].values.tolist())
    num_uniqueworkers =len(uni_workers)
    print('unique workers: ',num_uniqueworkers)
    ys = []
    for label in final_dff['label'].values.tolist():
        if label=='misleading':
             ys.append(1)
        else:
            ys.append(2)
    print(len(ys))
    iis = [item for sublist in [[b]*3 for b in range(1,int(len(final_dff)/3)+1)] for item in sublist]
    print(len(iis))
    org = final_dff['workerid'].values.tolist()
    temp = defaultdict(lambda: len(temp))
    jjs = [temp[ele]+1 for ele in org]
    temp_dict = dict(temp)
    print(len(jjs))
    schools_code = """
    data {
      int<lower=1> J; //number of annotators
      int<lower=2> K; //number of classes
      int<lower=1> N; //number of annotations
      int<lower=1> I; //number of items
      int<lower=1,upper=I> ii[N]; //the item the n-th annotation belongs to
      int<lower=1,upper=J> jj[N]; //the annotator which produced the n-th annotation
      int y[N]; //the class of the n-th annotation
    }

    transformed data {
      vector[K] alpha = rep_vector(1.0/K,K); //uniform prior(true label)
      vector[K] eta = rep_vector(10,K); // Dirichlet prior (A_ij)
    }

    parameters {
      simplex[K] epsilon[J]; // sum of epsilon_j =1 
      real<lower=0, upper=1> theta[J];
    }

    transformed parameters {
      vector[K] log_q_c[I];
      vector[K] log_alpha;

      log_alpha = log(alpha);

      for (i in 1:I) 
        log_q_c[i] = log_alpha;

      for (n in 1:N) 
      {
        for (h in 1:K)
        {
          int indicator = (y[n] == h);
          log_q_c[ii[n],h] = log_q_c[ii[n],h] + log( theta[jj[n]] * indicator + (1-theta[jj[n]]) * epsilon[jj[n],y[n]] );
        }
      }
    }

    model {
      for(j in 1:J)
      {
        epsilon[j] ~ dirichlet(eta);
        theta[j] ~ beta(0.5,0.5);
      }

      for (i in 1:I)
        target += log_sum_exp(log_q_c[i]);
    }

    generated quantities {
      vector[K] q_z[I]; //the true class distribution of each item

      for(i in 1:I)
        q_z[i] = softmax(log_q_c[i]);
    }
    """

    schools_data = {"J": num_uniqueworkers,
                    "y": ys,
                    "N": len(final_dff),
                    "K":2,
                    "I":int(len(final_dff)/3),
                    "ii":iis,
                    "jj":jjs,
                    }

    posterior = stan.build(schools_code, data=schools_data)
    fit = posterior.sample(num_chains=4, num_samples=1000)

    return fit,temp_dict
