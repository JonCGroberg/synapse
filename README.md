# Synapse
Machine learning in C#. Written from scratch with no dependencies

# Using Synapse
```
using Synapse;

double[][] input =
{
    new double[] { 0, 0 },
    new double[] { 0, 1 },
    new double[] { 1, 0 },
    new double[] { 1, 1 }
};
double[][] output =
{
    new double[] { 0 },
    new double[] { 1 },
    new double[] { 1 },
    new double[] { 0 }
};

Network network = new(2, 1);
network.Init(input, output);

WriteLine("\nEstimates:" + network.Predict());
WriteLine("\nError:"     + network.Error());
```
## Features
### Matrix math library 
  - [x] Addition/Subtraction
  - [x] Dot product 
  - [x] Scaler Multiplication/division
### Statistics Library 
  - [x] Normal Distribution (extension of Random)
### Functional Methods
  - [x] ZipWith 2D
  - [x] Map 2D
### Activation Functions
  - [ ] Sigmoid
  - [ ] Relu
### Network Architectures
  - [ ] Simple Feed Forward
  - [ ] CNN
  
