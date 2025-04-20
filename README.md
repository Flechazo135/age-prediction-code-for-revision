# age-prediction-code-for-revision

This study employed an **ensemble model** with 17 neural network (NN) classifiers for **age prediction** based on salivary DNA methylation. This repository provided the code and input data used in our study. The research was organized into four main sections, with the corresponding code and input data available in their respective folders.

**The following scripts were provided for various tasks**:
- **Randomly splitting the samples.py**: Randomly partitions the samples into training and testing sets (provided in Correlation and the ensemble model construction/Code/`Randomly splitting the samples.py`）.
- **The ensemble model (17-NN) with LIME.py**: Generates model explanations using **LIME** for interpretability for **Reviewer 3** (provided in Correlation and the ensemble model construction/Code/`The ensemble model (17-NN) with LIME.py`）.
- A **Residual Plot** was added to address **Reviewer 5's** comments, and the corresponding code can be found in **"Residual Plot.py"** (provided in Model comparison/Code/`Residual Plot.py`）.

### Overview of Sections:
1. **AR-CpGs selection**  
2. **Correlation and the ensemble model construction**  
3. **Model comparison**  
4. **Cross-platform**

> Note: The input data for AR-CpGs selection is too large to include directly and is stored on Google Drive:  
[Download AR-CpGs Input Data](https://drive.google.com/drive/folders/1Ns70dnPsp7Jnz_I8hY-MwYvepCTcyBZC?usp=drive_link).

All analyses were conducted using:
- **PyCharm Community Edition** (version 2023.3.3)
- **Python** (version 3.11)
- **scikit-learn** (version 1.4.0)

## AR-CpGs selection

The input data for this step could be found at:  
[Download AR-CpGs Input Data](https://drive.google.com/drive/folders/1Ns70dnPsp7Jnz_I8hY-MwYvepCTcyBZC?usp=drive_link).

This section demonstrated the complete procedure for **CpG sites selection**, accompanied by scripts for data visualization. The process of selecting CpG sites was as follows:

![AR-CpGs Selection](https://github.com/user-attachments/assets/7a2dfbde-5ab0-4250-995e-ad76c3d2bc5f)

## Correlation and the ensemble model construction

### Correlation Calculation

We first calculated the correlations of 10 CpG sites using **283 samples**. The corresponding code can be found in `Correlation.py`.

### Ensemble Model Construction

We constructed the ensemble model by assembling **N base classifiers**. Each classifier discretizes ages into categories with a bin width of N years, with subsequent classifiers shifting bins by one year to ensure comprehensive age range coverage. An example with **N=10** was illustrated below:

![Ensemble Model Example](https://github.com/user-attachments/assets/760e5449-e999-4757-8580-e942a9620506)

We evaluated the performance of **seven classification algorithms**:
- Linear Discriminant Analysis (LDA)
- Logistic Regression (LR)
- Decision Tree (DT)
- K-Nearest Neighbors (KNN)
- Naive Bayes (NB)
- Support Vector Machine (SVM)
- Neural Networks (NN)

We tested the ensemble model with **N** ranging from **1 to 53**. The results showed that the **neural network (NN)** algorithm performed best when **N=17**, so we selected this model as the final prediction model. The code can be found in the file **"The ensemble model with 17-NN.py"**.

The following scripts were provided for various tasks:
- **Randomly splitting the samples.py**: Randomly partitions the samples into training and testing sets.
- **The ensemble model (17-NN) with LIME.py**: Generates model explanations using **LIME** for interpretability.

## Model comparison

We compared our ensemble model with **five conventional regression models**, including:
- Multiple Linear Regression (MLR)
- Decision Tree Regression (DTR)
- K-Nearest Neighbors Regression (KNN regression)
- Support Vector Regression (SVR)
- Neural Network Regression (NN regression)

We also provided visualization scripts to clearly demonstrate the advantages of our ensemble model.

Additionally, we applied our model to data from **ref38** (code located in **"Comparison with those in previous studies.py"**), and the results showed that our model outperformed previous reports to some extent.

A **Residual Plot** was added to address **Reviewer 5's** comments, and the corresponding code can be found in **"Residual Plot.py"**.

## Cross-platform

To mitigate platform differences, we introduced **dummy variables**. Initially, a single dummy variable was used to handle data from two platforms. Subsequently, two dummy variables were employed for a combined analysis of data from three detection platforms.

---
