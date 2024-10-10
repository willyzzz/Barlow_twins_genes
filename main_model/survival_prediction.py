#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.metrics import concordance_index_censored
import scipy.cluster.hierarchy as sch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from lifelines import KaplanMeierFitter
np.random.seed(42)

path = '../Synapse_metabric/Clinical_Overall_Survival_Data_from_METABRIC.txt'
surv = pd.read_csv(path, index_col=0, sep=',')
surv


# In[2]:


pam50_path = '../Synapse_metabric/Complete_METABRIC_Clinical_Features_Data.txt'
pam50 = pd.read_csv(pam50_path, index_col=0, sep=',', on_bad_lines='skip')
pam50 = pam50[['NOT_IN_OSLOVAL_Pam50Subtype']]
pam50 = pam50.rename(columns={"NOT_IN_OSLOVAL_Pam50Subtype": "Pam50Subtype"})


# In[3]:


pam50


# In[4]:


bulk = torch.load('../result/Metabric_20241003_115435/bulk_tensor.pt', map_location=torch.device('cpu'))
bulk


# In[5]:


embedding = torch.load('../result/Metabric_20241003_115435/embedding.pt', map_location=torch.device('cpu'))
embedding


# In[6]:


from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def survival_analysis_with_lifelines(surv, bulk, test_size=0.2, random_state=42):
    """
    Perform survival analysis using lifelines with Cox Proportional Hazards model and Kaplan-Meier estimation.
    
    Parameters:
        surv (pd.DataFrame): DataFrame containing survival data with 'time' and 'status' columns.
        bulk (pd.DataFrame): DataFrame containing gene expression data.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed used by the random number generator.
    """
    # 1. Data Preprocessing: Select common samples and filter data
    common_samples = np.intersect1d(surv.index, bulk.index)
    surv = surv.loc[common_samples]
    bulk = bulk.loc[common_samples]

    # 2. Combine survival and expression data into one DataFrame for lifelines
    data = pd.concat([surv, bulk], axis=1)

    # 3. Split the data into training and testing sets
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    print(f"Training set size: {train_data.shape}, Test set size: {test_data.shape}")

    # 4. Fit Cox Proportional Hazards model on training set
    cph = CoxPHFitter()
    cph.fit(train_data, duration_col='time', event_col='status')
    
    # 5. Print model summary and coefficients
    print(cph.summary)

    # 6. Calculate C-index on training and testing sets
    train_c_index = cph.concordance_index_
    test_c_index = concordance_index(test_data['time'], -cph.predict_partial_hazard(test_data), test_data['status'])
    print(f"C-Index on Training Set: {train_c_index}")
    print(f"C-Index on Test Set: {test_c_index}")

    # 7. Predict and plot the survival function for a random sample from the test set
    random_sample = test_data.sample(n=1, random_state=random_state)
    surv_func = cph.predict_survival_function(random_sample)
    
    # Plot survival function for random sample
    plt.figure(figsize=(10, 6))
    surv_func.plot()
    plt.title("Survival Function for Random Test Sample")
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.show()

    # 8. Split test samples into high-risk and low-risk groups based on median predicted risk
    risk_scores = cph.predict_partial_hazard(test_data)
    risk_threshold = risk_scores.median()
    high_risk = test_data[risk_scores > risk_threshold]
    low_risk = test_data[risk_scores <= risk_threshold]

    # 9. Compute Kaplan-Meier survival curves for each group in the test set
    kmf_high = KaplanMeierFitter()
    kmf_high.fit(high_risk['time'], event_observed=high_risk['status'], label='High Risk (Test)')
    
    kmf_low = KaplanMeierFitter()
    kmf_low.fit(low_risk['time'], event_observed=low_risk['status'], label='Low Risk (Test)')

    # 10. Plot Kaplan-Meier survival curves
    plt.figure(figsize=(10, 6))
    kmf_high.plot()
    kmf_low.plot()
    plt.ylim(0, 1)
    plt.ylabel(r"Estimated probability of survival $\hat{S}(t)$")
    plt.xlabel("Time")
    plt.title("Kaplan-Meier Survival Curves (Test Set)")
    plt.legend()
    plt.show()


# In[7]:


def survival_analysis_with_pam50_lifelines(surv, pam50_labels):
    # 1. Data preprocessing: Ensure that the indices of surv and pam50_labels are aligned to find common samples
    if isinstance(pam50_labels, pd.DataFrame):
        if "Pam50Subtype" in pam50_labels.columns:
            pam50_labels = pam50_labels["Pam50Subtype"]
        else:
            raise ValueError("pam50_labels does not contain a column named 'Pam50Subtype'.")

    # 2. Align common samples
    common_samples = surv.index.intersection(pam50_labels.index)
    surv = surv.loc[common_samples]
    pam50_labels = pam50_labels.loc[common_samples]

    # 3. Encode the PAM50 labels into numerical values for Cox model input
    label_encoder = LabelEncoder()
    pam50_encoded = label_encoder.fit_transform(pam50_labels)

    # Create a DataFrame with the encoded PAM50 labels and survival data
    data = pd.DataFrame({
        'time': surv['time'],
        'status': surv['status'],
        'Pam50Subtype': pam50_encoded
    })

    # 4. Fit Cox Proportional Hazards model
    cph = CoxPHFitter()
    cph.fit(data, duration_col='time', event_col='status')

    # Print model summary
    print(cph.summary)

    # 5. Calculate C-index
    c_index = cph.concordance_index_
    print(f"C-Index: {c_index}")

    # Step 6: Plot Kaplan-Meier survival curves for each PAM50 subtype
    plt.figure(figsize=(10, 7))
    pam50_subtypes = label_encoder.classes_  # Get unique PAM50 subtypes

    for subtype in pam50_subtypes:
        # Filter the samples based on the current PAM50 subtype
        mask = pam50_labels == subtype  # Boolean mask
        kmf = KaplanMeierFitter()
        kmf.fit(data.loc[mask, 'time'], event_observed=data.loc[mask, 'status'], label=subtype)

        # Plot the survival curve for the subtype
        kmf.plot_survival_function()

    # Set labels and title
    plt.ylim(0, 1)
    plt.ylabel(r"Estimated probability of survival $\hat{S}(t)$")
    plt.xlabel("Time")
    plt.title("Kaplan-Meier Survival Curves for Different PAM50 Subtypes")
    plt.legend()
    plt.show()


# In[9]:


survival_analysis_with_pam50_lifelines(surv, pam50)


# In[12]:


survival_analysis_with_lifelines(surv, bulk)


# In[ ]:


survival_analysis_with_lifelines(surv, embedding)


# In[ ]:




