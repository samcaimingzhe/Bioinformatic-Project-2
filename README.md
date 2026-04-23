# TabPFN note

### What is ML doing in prediction task?
A dataset with a bunch of complete feature columns and 1 single label column with missing value. 
Filling the missing values of a label column depending on rest of the feature columns.

TabPFN is designed for handing small-to medium-sized datasets with up to 10,000 samples and 500 features. So for lager datasets and highly non-smooth regression datasets, traditional ML approach such as CatBoost, XGB or AutoGluon performs better than TabPFN.

### Generated dataset: how does these data generated, they must base on some rules? What kind of rules?

Imagine in the real world, we have a lot of cause-effects, which is exist everywhere, and we observe data from them, but we couldn’t know the exact cause-effects patterns. So in TabPFN, we want to simulate this process. If the real world can generate cause-effects and we can observe data from them, then we can also use structural causal model to generate casual data. These data are reasonable because they have cause-effects relationship.

### Complexity
TabPFN is cell-based attention mechanism: 

$$O(nm^2+n^2m)$$

### Synthetic prior datasets
TabPFN: structural causal models (SCMs)
TabICL and TabForestPFN: mixed tree-based priors to inject tree inductive biases.

## Method
### Graph structure sampling
The SCMs underlying each generated dataset are based on DAG. These graphs are sampled by using the **growing network with redirection sampling method**. This may generate random scale-free networks through a preferential attachment process.

> scale-free network: 
> preferential attachment process: 

In graph or network, node represent variables (features), and edge represent the interaction between nodes. More edges means more condense network.
The graph size is determined by the number of nodes N, and the complexity is determined by the number of edges. The N is sampled from a log-uniform distribution with hyperparameter a and b.
logN~U(a,b)
