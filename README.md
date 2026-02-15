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

## **5.Overall results and Weekly improvemnents** 

<img width="1243" height="652" alt="image" src="https://github.com/user-attachments/assets/66fbf3ed-d192-4059-a572-144498165112" />

## **6.1 Datasheet and model card** 
## **6.1 Datasheet** 
**Overview and Motivation**

This project applies Bayesian Optimisation (BO) to maximise the outputs of a set of unknown (black-box) functions under realistic constraints. Each function represents a simulated real-world optimisation problem where the internal structure is hidden and only input–output observations are available. A key constraint is that only one query per function can be submitted each week, with delayed feedback, requiring careful balancing between exploration and exploitation. The dataset and modelling framework were created by Cristina Donadoni as part of a capstone project, with no external funding. The aim is to study sequential decision-making when evaluations are costly, slow and opaque, closely mirroring real-world optimisation settings.

**Dataset Composition and Collection**

The dataset consists of eight independent black-box optimisation problems, with input dimensionality ranging from 1D to 8D. Each data instance corresponds to a weekly query (input vector) and its observed scalar output. Data is collected incrementally over ten weekly rounds, using a deterministic, adaptive sampling strategy driven by Bayesian Optimisation acquisition functions. The dataset is complete relative to the evaluation budget but not exhaustive of the underlying function space. All data is numerical and synthetic, with no personal, sensitive or offensive content. There are no missing values and no fixed train/test splits, as models are retrained each week using all available observations.

**Preprocessing and Data Handling**

Preprocessing varies by function and includes input scaling and standardisation, output transformations (e.g. log, signed root, Yeo–Johnson), and outlier handling where required (e.g. One-Class SVM). Automatic relevance determination (ARD) is used to infer input importance. Raw observations are preserved alongside transformed versions to support reproducibility.

**Intended Uses and Limitations (Dataset)**

The dataset is intended for educational and experimental use, particularly for studying Bayesian Optimisation under constrained evaluation budgets. It is not suitable for fairness analysis, real-time decision systems or direct high-stakes deployment without domain-specific validation. Results are sensitive to early observations, kernel choice and modelling assumptions.
## **6.2 Model Card** 
**Model Overview**

Name: Adaptive GP-Based Bayesian Optimisation Framework
Type: Sequential decision-making using Gaussian Processes
Version: v1.0
This is a decision framework, not a single predictive model, designed to propose optimal query points under uncertainty.

**Intended Use**

The framework is suitable for black-box optimisation with costly or delayed evaluations in continuous input spaces. It is not designed for discrete optimisation or environments requiring interpretability of the underlying function.

**Model Details and Strategy**

Across ten rounds, the approach uses Gaussian Processes with Matérn or RBF kernels (with ARD), combined with adaptive acquisition functions. Strategies include UCB, Expected Improvement (EI), and hybrid EI/UCB or PI/UCB approaches. Candidate points are generated via dense grids (low-dimensional problems) or constrained random sampling (higher dimensions). The strategy evolves per function, reflecting learning from observed performance and uncertainty.

**Performance Summary**

Performance is evaluated using best observed output, percentage improvement from baseline, and speed of convergence. Improvements range from ~3% to over 200%, with several functions achieving their best results by Week 6, demonstrating effective exploration–exploitation trade-offs under tight query constraints.

**Assumptions, Limitations and Ethics**

The approach assumes relatively smooth, stationary functions and reasonable signal-to-noise ratios. Limitations include sensitivity to kernel misspecification and the high cost of poor early exploration decisions. Transparency in documenting transformations, acquisition logic and assumptions supports reproducibility, auditability and responsible real-world adaptation. Additional detail beyond this model card would add complexity without materially improving clarity for the intended audience.
