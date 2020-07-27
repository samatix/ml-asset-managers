# Machine Learning for Asset Managers Toolbox 
---

## Introduction

I started this work with the personal goal to master and test the concepts described in the book [Machine Learning for Asset Managers](https://www.cambridge.org/core/books/machine-learning-for-asset-managers/6D9211305EA2E425D33A9F38D0AE3545).

This repository contains the implementation of these concepts and answers to the exercises. 

## Disclaimers
- This work is licensed under the Apache License Version 2.0 unless specified otherwise. 

---

## Getting Started

Clone this repository locally: 

```
git clone https://github.com/samatix/ml-asset-managers.git
``` 

---

## Available Utilities

### Entropy Calculation
- Initial inputs
```python
import numpy as np 

from src import entropy

x = np.array([
            -1.068975469981432, 0.37745946782651796, -1.4503714157560206,
            -2.0189938521856945, -0.6720045848322777, 1.0585123584971843,
            0.10590926320793637, 2.8321554887980236, -1.6415040483953403,
            0.8256354839964547
        ]) 

e = np.array([
            -0.4355421091328046, 0.08072721876416557, -0.18228820347023844,
            0.1553520158613207, -0.07595958194802123, -1.5300711428677072,
            -1.482275653452137, -0.035086362949407486, -1.3101091248694603,
            -0.7693024441943448
        ])
```


- Calculate the marginal entropy 
```python
marginal = entropy.marginal(x, bins=10)
assert marginal == 1.8866967846580784
```
- Calculate the join entropy
```python
joint = entropy.joint(x, e)
assert joint == 1.8343719702816235
```
- Calculate the mutual information (normalized as well)
When the variables are independent
```python
# Independent Variables
y = 0 * x + e
mi = entropy.mutual_info(x, y, bins=5)
nmi = entropy.mutual_info(x, y, bins=5, norm=True)
corr = np.corrcoef(x, y)[0, 1]

# No correlation and normalized mutual information is low (small
# observations set)
assert corr == -0.08756232304451231
assert nmi == 0.4175336691560972
```

When the variables are linearly correlated 
```python
# Linear Correlation
y = 100 * x + e
nmi = entropy.mutual_info(x, y, bins=5, norm=True)
corr = np.corrcoef(x, y)[0, 1]

# Linear correlation between x and y both the correlation and
# normalized mutual information are close to 1
assert corr == 0.9999901828471118
assert nmi, 1.0000000000000002

```

When the variables are non-linearly correlated 

```python
# Linear Correlation
y = 100 * abs(x) + e
nmi = entropy.mutual_info(x, y, bins=5, norm=True)
corr = np.corrcoef(x, y)[0, 1]

# Non linear correlation between x and y. Correlation is close to 0
# but the normalized mutual information detects correlation betweeen x and y
assert corr == 0.13607916658759206
assert nmi, 1.0000000000000002

```

- Calculate the conditional entropy 
````python
conditional = entropy.conditional(x, e)

# H(X) >= H(X|Y)
assert entropy.marginal(x) >= conditional

# H(X|X) = 0
assert entropy.conditional(x, x) == 0

assert conditional == 0.8047189562170498
````


- Calculate the variation of information (normalized as well) (similar to the previous example of mutual information)

- The possibility to use the optimal number of bins when calling the previous functions with bins=None

```python
numb_bins = entropy.num_bins(n_obs=10)
assert numb_bins, 3

numb_bins = entropy.num_bins(n_obs=100)
assert numb_bins, 7

# For joint entropy with zero correlation
numb_bins = entropy.num_bins(n_obs=10, corr=0)
assert numb_bins, 3

# For joint entropy with total correlation
numb_bins = entropy.num_bins(n_obs=10, corr=1)
# In this case, we return the num_bins using corr=None 

# For joint entropy with 0.5 correlation
numb_bins = entropy.num_bins(n_obs=10, corr=0.99)
assert numb_bins, 7
```

### Fixtures 
- Generate a random block correlation matrix

```python
from src.fixtures import CorrelationFactory

cf = CorrelationFactory(
                        n_cols=10,
                        n_blocks=4,
                        sigma_b=0.5,
                        sigma_n=1,
                        seed=None
                    )

corr = cf.random_block_corr()
```


### Notebooks 
- Chapter 2 on distances summary and exercises tentative solutions 


## Bibliography 
- López de Prado, M. (2020). Machine Learning for Asset Managers (Elements in Quantitative Finance). Cambridge: Cambridge University Press. doi:10.1017/9781108883658