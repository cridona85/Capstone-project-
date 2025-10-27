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
