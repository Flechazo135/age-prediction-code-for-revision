# age-prediction-code-for-revision
This study employed an ensemble model with 17 neural network (NN) classifiers for age prediction based on salivary DNA methylation. This repository provided the code and input data used in our study. Our research was organized into four main sections, with the corresponding code and input data available in their respective folders (noting that the input data for AR-CpGs selection was too large to include directly and had therefore been stored in "https://drive.google.com/drive/folders/1Ns70dnPsp7Jnz_I8hY-MwYvepCTcyBZC?usp=drive_link"). Furthermore, we provided additional descriptions for the code and data to enhance their clarity and facilitate use by other researchers.
All the analyses were conducted using PyCharm Community Edition software (version 2023.3.3) and Python (version 3.11). And the "scikit-learn" package (version 1.4.0) was used for all models.
# AR-CpGs selection
Input Data could be found in "https://drive.google.com/drive/folders/1Ns70dnPsp7Jnz_I8hY-MwYvepCTcyBZC?usp=drive_link".
The code demonstrated the complete procedure for site selection, accompanied by scripts for data visualization.
The process of selecting CpG sites was as follows:
![export](https://github.com/user-attachments/assets/7a2dfbde-5ab0-4250-995e-ad76c3d2bc5f)
# Correlation and the ensemble model construction
We first calculated the correlations of 10 methylation sites using 283 samples, with the corresponding code provided in “Correlation.py”
Next, we constructed the following ensemble model. The ensemble model is constructed by assembling N base classifiers, each discretizing ages into categories with a bin width of N years, with subsequent classifiers shifting bins by one year to ensure comprehensive age range coverage (As illustrated in the figure, N=10 was presented as an example)：
![2](https://github.com/user-attachments/assets/760e5449-e999-4757-8580-e942a9620506)
We evaluated the performance of seven classification algorithms (Linear Discriminant Analysis (LDA,), Logistic Regression (LR), Decision Tree (DT), K-nearest Neighbors (KNN), Naive Bayes (NB), Support Vector Machine (SVM) and NN) for N ranging from 1 to 53 ("Each algorithm with N in the ensemble model.py". The results indicated that the neural network algorithm performed best when N=17, and thus, this model was selected as our final prediction model ("The ensemble model with 17-NN.py").
Specifically, "Randomly splitting the samples.py" was used to randomly partition the samples into training and testing sets; "The ensemble model (17-NN) with LIME.py" was used to generate model explanations using LIME (for Reviewer 3).
# Model comparison
Here, we compared our ensemble model with five other conventional regression models (including multiple linear regression (MLR), decision tree regression (DTR), K-nearest neighbors regression (KNN regression), SVR and neural network regression (NN regression) as LDA and NB algorithms were not suitable for regression analysis involving continuous variables). We also provided the corresponding visualization scripts to clearly demonstrate the advantages of our ensemble model.
Furthermore, we applied our model to the data from ref38 ("Comparison with those in previous studies.py"). The results demonstrated that our model outperformed previous reports to some extent.
Specifically, "Residual Plot.py" was added specifically to address Reviewer 5's comments.
# Cross-platform
In this context, we introduce dummy variables to mitigate platform differences. Initially, a single dummy variable was used to handle data from two platforms. Subsequently, two dummy variables were employed for a combined analysis of data from three detection platforms.
