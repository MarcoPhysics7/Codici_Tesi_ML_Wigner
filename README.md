
Created on Wed Jul  2 11:47:21 2025
@author: marcoascari

The repository contins all the code I developed for my bachelor thesis.
It's the framework for the study of image classification about Wigner quasiprobability distrubutions for Fock states superpositions.
Code organization:
- dataset_winger_pulito -> creates from scratch the dataset of the Wigner functions as 2D images.
- dataset_wigner_noisy_grid -> creates noisy(blurred) images to test the robustness of Machine Learning models I used.
- dataset_mixed -> creates the images of mixed quantum states (original states mixed with vacuum) to simulate Quantum decoherence in first approximation.
- files Nameofthemodel_SEARCH -> used for hyperparameter optimization (can be improved soon). I used a grid search. Training and validation processes are done in this code.
- files TestNameofthemodelRumore -> test the models on blurred and mixed Wigner functions.

Machine Learning models used: Multilayer Perceptron, 2D convolutional neoural networks, Random Forest


