# General Multilayer Perceptrons
A hobby project implementing a framework for creating and training arbitrary multilayered perceptrons with different learning strategies.

---
## Table of Contents
1. [**Motivation**](#motivation)
2. [**Usage**](#usage)
3. [**License**](#license)
---
## Motivation
After reading the book "Computational Intelligence" by Rudolf Kruse and coauthors, I set myself the goal to implement a functional, learning multilayered perceptron. After implementation of an arbitrary layered perceptron with basic gradient descend, I abstracted the classes further and eventually implemented all typed of gradient descend mentioned in the book, as well as sensitivity-analysis of the input-neurons. The terminology used for the naming and documentation was made to closely mirror the book, and all the theory used is explained by the book.
## Usage
First a perceptron of desired dimensions with the prefered technique of gradient-descend must be created, as well as a compatible learning-bath consisting of multiple input-vectors with the desired output-vectors. Then the network can be trained either via batch-or-onlinelearning for arbitrary many iterations, and after that the complete state of all weights and biases in the trained network can bei displayed in console. The total squared error and the sensitivity of individual input neurons over the learning-batch can also be displayed, and outputs for data outside the training-batch can be gained via ForwardPropagation. As an example a simple neural network is trained on the binary operation XOR for 100 iterations of batch-learning, and the final state displayed in console when running the program.
## License
This project is licensed under the terms of the MIT license.



