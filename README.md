# Non-Linear Geometric Classifier (MLP)
A Multi-Layer Perceptron (MLP) implemented in C++ using the Eigen library. This project demonstrates the ability of neural networks to learn complex, non-linear spatial boundaries—specifically identifying whether a point $(x, y)$ 
falls within a circular area.
## The Task
### Circle Classification
The network is trained to classify points in a 2D plane. The target boundary is defined by the inequality:
$$x^2 + y^2 \leq 0.5$$
Points inside the radius of 
$\sqrt{0.5}$ 
are labeled as 1 (Inside), while points outside are labeled as 0 (Outside). This is a classic "non-linearly separable" problem that a single-layer perceptron could not solve.
## Neural Network Architecture
The model uses a feedforward architecture with backpropagation:
### Input Layer
2 neurons (representing $x$ and $y$ coordinates).
### Hidden Layer
4 neurons with Sigmoid activation, allowing the model to project inputs into a higher-dimensional space to find a non-linear boundary.
### Output Layer
1 neuron (representing the probability of the point being "Inside").
## Logic Components
### Layer Class 
Manages weights, biases, and stores state (last_input/last_output) for gradient calculations.
### Forward Pass
Uses the Sigmoid activation function $\sigma(z) = \frac{1}{1+e^{-z}}$.
### Backpropagation
Implements stochastic gradient descent to minimize the error by adjusting weights based on the chain rule.
## Mathematical Implementation
The weight update $\Delta w$ is calculated using the delta rule:
$$\Delta w = \eta \cdot (\text{target} - \text{output}) \cdot \sigma'(z) \cdot \text{input}$$Where:$\eta$ 

(eta): Learning rate (set to 0.1).$\sigma'(z)$: The derivative of the sigmoid function, calculated as $output \cdot (1 - output)$.🚀 How it Works (Main Flow)Training Phase: The program generates 1,000 random points within a $[-1, 1]$ range. It checks them against the circle equation and trains the MLP.User Interaction: After training, the program prompts the user to enter a custom $(x, y)$ coordinate.Inference: The model predicts the probability. If the output is $> 0.5$, the point is classified as "Inside the Circle".🛠️ Technical DependenciesEigen Library: Used for all matrix and vector operations.Standard C++ Random: Utilizes std::mt19937 and std::normal_distribution for weight initialization and data generation.
