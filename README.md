# **Capstone Project - building and optimising an ML model within a simulated black-box environment**
## **1.Project overview**
This capstone project forms part of the Imperial College London AI & Machine Learning programme and is designed to bridge the gap between theory and practice by applying ML techniques to a realistic optimisation challenge. The project simulates a real-world, black-box environment, where the internal workings of the problem are unknown and information is limited, mirroring the uncertainty often faced by data scientists and ML practitioners in applied settings.

In this challenge, learners are tasked with optimising unknown functions using limited data and feedback. These black-box functions represent real-world applications such as radiation detection, robotic control where evaluations are costly or constrained. The aim is not to find a perfect solution but to demonstrate a sound, evidence-based, and iterative optimisation process. Each week, new data points are queried, model predictions are updated and strategies are refined.
## **2. Inputs and output**
The descriptions below highlight the key properties of each function and provide examples of where similar optimisation challenges occur in practice. The initial input (x) and output (y) data are supplied as .npy files (the standard binary file format used by the NumPy library in Python), and new input and output data are provided each week.

<img width="1229" height="871" alt="image" src="https://github.com/user-attachments/assets/93cf30f2-c1ff-47e8-9d70-62991f481eaa" /><img width="1231" height="662" alt="image" src="https://github.com/user-attachments/assets/9f4c282f-62a2-4889-887a-4fd06aebdd52" />
## **3. Challenge objectives** 
The objective of the project is to apply Bayesian Optimisation to identify input values that are most likely to maximise the output of a set of unknown (black-box) functions. Each function represents a simulated real-world optimisation problem where the internal structure is hidden and only input–output observations are available.

The goal is therefore to iteratively propose new query points that improve model performance and move closer to the global maximum for each function. A key constraint is that only one query per function can be submitted each week, which limits the speed of learning and requires careful balancing between exploration (testing new regions of the input space) and exploitation (focusing on regions expected to yield higher outputs).

Additional limitations include the unknown structure of each function and a response delay between query submission and receiving results. These constraints mirror real-world conditions in which evaluations are costly or time-consuming, and the underlying process cannot be directly observed.
## **4. Approach** 
Across the first three iterations of the BBO capstone project, my approach has evolved from broad exploration to more targeted optimisation while maintaining a balance between exploration and exploitation.In the first iteration, I employed Bayesian Optimisation with a Gaussian Process (GP) surrogate model, using RBF and Matern kernels with and without ARD and WhiteKernel components. My strategy initially emphasised exploration to understand the behaviour of each unknown function and progressively incorporated exploitation as clearer trends emerged. The Expected Improvement (EI) acquisition function guided the selection of new query points for all the functions in week 1, ensuring a structured trade-off between sampling uncertain regions and focusing on promising areas.

I prioritised exploratory learning—visualising data, testing kernel combinations and applying transformations or standardisation where outputs showed large variation (e.g. Function 5). In Week 2, I focused on stabilising the model through reproducibility checks, improved scaling and consistency across iterations. This allowed the GP to form a more robust surrogate model without premature convergence. By Week 3, I increased the number of candidate points for higher-dimensional functions (from 5,000 to 50,000) and began tuning kernel hyperparameters such as length-scales and noise variance to adapt the GP to different data structures.

While I have mainly relied on Bayesian techniques, I recognise that Support Vector Machines (SVMs) could complement the GP by classifying high- vs low-performing regions, particularly using a soft-margin or kernel-based SVM to handle noisy, non-linear boundaries. As data accumulates, some overfitting has appeared in low-dimensional functions, highlighting the need for smoother priors and possible dimensionality reduction.

Overall, this iterative, black-box process mirrors real-world ML challenges—requiring critical thinking under uncertainty, evidence-based decision-making and adaptive model refinement to progressively move toward optimal solutions.

## **6. Observations as per week 4** 
### Strategy and “Support Vectors”

In this optimisation task, support vectors refer to input points that lie near a decision boundary or a region of rapid change in the response surface. Recognising these points helps identify areas where the function output shifts quickly, indicating where the next query should explore to capture new information. Across the iterations, I achieved improvements from the original outputs for all functions except Function 1, with percentage improvements ranging between 3% (Function 8) and 109% (Function 4). In particular:

Function 4 showed consistent improvement each week, with the highest gain of 109% in Week 4, indicating a well-aligned exploration path.
Functions 3, 5, and 7 exhibited steep output transitions (improvements of 68%, 67%, and 88% respectively) between Weeks 2 and 3. These regions likely acted as support vectors — areas of rapid change but the subsequent decline suggests potential over-exploitation near a local optimum.
Functions 2, 6, and 8 showed more moderate improvements (9%, 6%, and 3%, respectively), with gains achieved between Weeks 1 and 3.
Querying around these rapidly changing input regions could help refine the surrogate model further and uncover additional optima.

Across all eight functions, I applied Bayesian Optimisation using Gaussian Processes (GPs) with different kernels and acquisition functions. My primary goal was to balance exploration and exploitation.

### Neural Networks and Gradients

I did not train a neural network surrogate, primarily because the available data was too limited to support robust training and because I have not yet fully completed the training material needed to apply it confidently. Neural networks generally require a large number of observations to generalise effectively whereas Gaussian Processes perform well with sparse data.

However, in higher-dimensional settings or with larger datasets, a neural network surrogate could capture more complex non-linear interactions. I plan to explore this approach in future iterations if appropriate. Examining the trends, Functions 3, 4, 6, and 8 displayed stronger sensitivity to specific input variables, as shown by significant week-to-week variation in outputs. In a neural surrogate context, I believe these would correspond to inputs with the steepest gradient magnitudes — indicating where small input changes cause large output differences. Recognising these inputs would help focus future experiments on the most influential variables, thereby accelerating convergence towards the global optimum.

### Classification

If the optimisation problem were reframed as a classification task, outcomes above a given performance threshold could be labelled as “good” and those below as “bad.”

Logistic regression or Support Vector Machines (SVMs) could then be used to estimate the decision boundary.
SVMs could be suitable since their support vectors lie along the margin,  aligning with the idea of focusing queries near regions of uncertainty or rapid change.
The main trade-off would be between exploitation/misclassification risk (staying close to known regions) and exploration (testing uncertain areas that could reveal better optima).

Model Selection and Interpretability

### So far, I believe Gaussian Processes have been the most appropriate model for these experiments, offering a good balance between exploration and exploitation. Linear regression lacked the flexibility to capture non-linear curvature. Neural networks would have required substantial tuning and carried a higher risk of overfitting given the limited dataset.
