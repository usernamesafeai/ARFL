# Adversarially Robust Feature Learning (ARFL)

## Overview

This repository contains the code and implementation for Adversarially Robust Feature Learning (ARFL). The project aims to 1)improve adversarial robustness of deep learning models against adversarial attacks and 2)maintain model performance on standard/clean data.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)

## Installation

To clone and run this repository, you'll need Git and Python installed on your computer. From your command line:

```bash
# Clone this repository
git clone https://github.com/usernamesafeai/ARFL.git

# Navigate into the repository
cd ARFL 

# Install dependencies
pip install -r requirements.txt


```

## Usage
### Experiment replication 
Step 1. 
Download the CMMD dataset into the folder named data or create your own synthetic two-moon dataset
For the CMMD dataset, you can [download it here](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230508).

Step 2. 
Configure the path to your dataset in the file below
```
experiments/CMMD/dual_adv_train_with_arfl.py

```
Step 3. run the code
```
python dual_adv_train_with_arfl.py

```

### Customized usage 
plugin the reularization term below to your loss function
 
```
from utils import calculate_robustness

reg = calculate_robustness(y_combined, features)
ce = criterion(outputs, y_combined)
loss = ce - gamma * reg
```

