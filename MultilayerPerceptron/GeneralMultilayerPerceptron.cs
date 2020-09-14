using System;
using System.Collections.Generic;
using System.Globalization;

namespace MultilayerPerceptron
{
    /// <summary>
    /// Implements a mulitlayer perceptron of variable layer count, with the logistic function as activation function for all
    /// neurons and the idendity as output function, trains with weight-decay
    /// </summary>
    abstract class GeneralMultilayerPerceptron
    {
        protected Vector[] biases;
        protected Matrix[] weights;
        protected Vector[] deltaTerms;
        protected Matrix[] currentWeightGradients;
        protected Vector[] currentBiasGradients;
        protected Vector[] outputs;
        private Function logisticFunction;
        private double weightPunishmentFactor;

        /// <summary>
        /// Fills the starting weights and biases of the multilayer perceptron with uniformely random values
        /// between the lower and upper bound
        /// </summary>
        /// <param name="hiddenNeuronLayerDimensions">The amount of neurons per hidden-layer, 
        /// starting with the hidden-layer after the input-layer</param>
        /// <param name="lowerBound">The lower bound for the random starting values</param>
        /// <param name="higherBound">The lower bound for the random starting values</param>
        public GeneralMultilayerPerceptron(int inputNeuronLayerDimension, ICollection<int> hiddenNeuronLayerDimensions, int outputNeuronLayerDimension,
            double lowerBound, double higherBound, double weightPunishmentFactor)
        {
            int neuronLayerCount = hiddenNeuronLayerDimensions.Count + 2;
            // Initialize neuron layers
            int[] neuronLayers = new int[neuronLayerCount];
            neuronLayers[0] = inputNeuronLayerDimension;
            hiddenNeuronLayerDimensions.CopyTo(neuronLayers, 1);
            neuronLayers[neuronLayerCount - 1] = outputNeuronLayerDimension;
            // Check wether all given layer-dimensions are greater or equal 1
            foreach (var layerDimension in neuronLayers)
            {
                if (layerDimension < 1)
                {
                    throw new ArgumentException("Dimension of each neuron layer must be greater or equal to 1");
                }
            }
            // Check wether the lower-bound is really lower than the higher-bound
            if (lowerBound > higherBound)
            {
                throw new ArgumentException("Given lower bound must be lower than the higher bound");
            }
            // Check wether weight punishment factor is between 0 and 1
            if (weightPunishmentFactor >= 1 || weightPunishmentFactor < 0)
            {
                throw new ArgumentException("Weight punishment factor must be between 0 and 1");
            }
            // Initialize fill with random values function
            Random rng = new Random();
            double difference = higherBound - lowerBound;
            Function randomValue = input => lowerBound + difference * rng.NextDouble();
            // Initialize and fill starting biases and weights with random values
            Vector[] randomStartingBiases = new Vector[neuronLayerCount - 1];
            Matrix[] randomStartingWeights = new Matrix[neuronLayerCount - 1];
            int previousNeuronLayerDimension = neuronLayers[0];
            int currentNeuronLayerDimension;
            for (int i = 0; i < neuronLayerCount - 1; i++)
            {
                currentNeuronLayerDimension = neuronLayers[i + 1];
                randomStartingBiases[i] = new Vector(currentNeuronLayerDimension);
                randomStartingWeights[i] = new Matrix(currentNeuronLayerDimension, previousNeuronLayerDimension);
                // Fill the starting biases and weights with random values according to the given bounds
                randomStartingBiases[i] = randomStartingBiases[i].ApplyFunctionOnAllComponents(randomValue);
                randomStartingWeights[i] = randomStartingWeights[i].ApplyFunctionOnAllComponents(randomValue);
                previousNeuronLayerDimension = currentNeuronLayerDimension;
            }
            InitializeNetwork(randomStartingBiases, randomStartingWeights, weightPunishmentFactor);
        }

        public GeneralMultilayerPerceptron(Vector[] startingBiases, Matrix[] startingWeights, double weightPunishmentFactor)
        {
            InitializeNetwork(startingBiases, startingWeights, weightPunishmentFactor);
        }

        /// <summary>
        /// Initializes all matrices and vectors needed for forward-propagation
        /// </summary>
        /// <param name="startingBiases">The starting values for the biases</param>
        /// <param name="startingWeights">The starting values for the weights</param>
        /// <param name="weightPunishmentFactor">The factor by which the weights should be reduced,commonly between 0.005 and 0.03</param>
        private void InitializeNetwork(Vector[] startingBiases, Matrix[] startingWeights, double weightPunishmentFactor)
        {
            int neuronLayerCount = startingBiases.Length + 1;
            this.weightPunishmentFactor = 1 - weightPunishmentFactor;
            // The logistic function applied on all vector components
            logisticFunction = input => 1 / (1 + Math.Exp(-input));
            // Initialize biases
            biases = startingBiases;
            // Initialize weights
            weights = startingWeights;
            // Initialize gradients
            currentWeightGradients = new Matrix[weights.Length];
            currentBiasGradients = new Vector[biases.Length];
            for (int i = 0; i < neuronLayerCount - 1; i++)
            {
                // Initialize weight gradients
                currentWeightGradients[i] = new Matrix(weights[i].Height, weights[i].Width);
                //Initialize bias gradients
                currentBiasGradients[i] = new Vector(biases[i].Dimension);
            }
            deltaTerms = new Vector[neuronLayerCount - 1];
            outputs = new Vector[neuronLayerCount];
        }

        public int NeuronLayerCount { get => outputs.Length; }
        public int InputDimension { get => weights[0].Width; }
        public int OutputDimension { get => biases[biases.Length - 1].Dimension; }
        public Vector Output { get => outputs[outputs.Length - 1]; private set => outputs[outputs.Length - 1] = value; }
        public Vector Input { get => outputs[0]; private set => outputs[0] = value; }
        private Vector OutputDeltaTerm { get => deltaTerms[deltaTerms.Length - 1]; set => deltaTerms[deltaTerms.Length - 1] = value; }

        /// <summary>
        /// Online learning, i.e. adapt the weights and biases for each learning task seperatly
        /// </summary>
        /// <param name="learningTask">The given learning tasks to train</param>
        public void OnlineLearning(LearningTask learningTask)
        {
            if (IsLearningTaskCompatible(learningTask))
            {
                foreach (var learningExample in learningTask)
                {
                    ForwardPropagation(learningExample.Input);
                    BackwardPropagation(learningExample.Output);
                    AdaptWeightsAndBiases();
                    PunishWeightsAndBiases();
                    ResetWeightAndBiasGradients();
                }
            }
            else
            {
                throw new ArgumentException("Learning task not compatible with multilayer perceptron");
            }


        }

        /// <summary>
        /// Batch learning, i.e. sum all the weight and bias changes for all learning examples and then
        /// adapt the weights and biases
        /// </summary>
        /// <param name="learningTask">The given learning tasks to train</param>
        public void BatchLearning(LearningTask learningTask)
        {
            if (IsLearningTaskCompatible(learningTask))
            {
                foreach (var learningExamples in learningTask)
                {
                    ForwardPropagation(learningExamples.Input);
                    BackwardPropagation(learningExamples.Output);
                }
                AdaptWeightsAndBiases();
                PunishWeightsAndBiases();
                ResetWeightAndBiasGradients();
            }
            else
            {
                throw new ArgumentException("Learning task not compatible with multilayer perceptron");
            }
        }

        /// <summary>
        /// Checks wether input- and output-dimensions of the learning-task equal the input- and
        /// outputlayer-dimensions of the multilayer perceptron
        /// </summary>
        /// <param name="learningTask">The learning task to test for compatibility</param>
        /// <returns>Is learning tsdk compatible</returns>
        private bool IsLearningTaskCompatible(LearningTask learningTask)
        {
            return learningTask.InputDimension == InputDimension && learningTask.OutputDimension == OutputDimension;
        }

        /// <summary>
        /// Propagates the given input through the neural network
        /// </summary>
        /// <param name="input">The given input to propagate</param>
        /// <returns>the output of the neural network for the given input</returns>
        public Vector ForwardPropagation(Vector input)
        {
            Input = input;
            for (int i = 0; i < NeuronLayerCount - 1; i++)
            {
                outputs[i + 1] = (weights[i] * outputs[i] - biases[i]).ApplyFunctionOnAllComponents(logisticFunction);
            }
            return Output;
        }

        /// <summary>
        /// Propagates the error according to the given desired output after forward-propagation through
        /// the network, adds the gradients to the gradnient matrices and vectors, does not adapt the weights
        /// or biases
        /// </summary>
        /// <param name="desiredOutput">The desired output for the last forward propagation</param>
        public void BackwardPropagation(Vector desiredOutput)
        {
            CalculateDeltaTerms(desiredOutput);
            AddWeightAndBiasGradients();
        }

        protected abstract void AdaptWeightsAndBiases();
        protected abstract void ResetWeightAndBiasGradients();

        /// <summary>
        /// Punishes large weights and biases by taking a fraction of them for
        /// each learning cycle
        /// </summary>
        private void PunishWeightsAndBiases()
        {
            for (int i = 0; i < NeuronLayerCount - 1; i++)
            {
                weights[i] *= weightPunishmentFactor;
                biases[i] *= weightPunishmentFactor;
            }
        }

        /// <summary>
        /// Adds the weight and bias gradients according to the delta-term to the 
        /// gradient vectors or matrices
        /// </summary>
        private void AddWeightAndBiasGradients()
        {
            for (int i = 0; i < NeuronLayerCount - 1; i++)
            {
                // Add weight gradients
                // For all neurons of the succeeding layer
                for (int j = 0; j < weights[i].Height; j++)
                {
                    // Change the corresponding rowvector in the delta-weight matrix according to the
                    // delta-term and the input (i.e. output of previous layer)
                    currentWeightGradients[i][j] -= deltaTerms[i][j] * outputs[i];
                }
                // Add bias gradients
                currentBiasGradients[i] += deltaTerms[i];
            }
        }

        /// <summary>
        /// Calculates the delta-terms for backpropagation
        /// </summary>
        /// <param name="desiredOutput">The desired output of the neural network</param>
        protected void CalculateDeltaTerms(Vector desiredOutput)
        {
            OutputDeltaTerm = (desiredOutput - Output).ComponentSum() * OutputDerivative(Output);
            // Delta terms form last hidden layer to input layer
            for (int i = deltaTerms.Length - 1; i > 0; i--)
            {
                deltaTerms[i - 1] = Vector.SchurProduct(weights[i].Transponse() * deltaTerms[i], OutputDerivative(outputs[i]));
            }
        }

        /// <summary>
        /// Calculates the derivative of the logistic function for a neuron layer with the output
        /// </summary>
        /// <param name="output">The output of the neuron layer</param>
        protected virtual Vector OutputDerivative(Vector output)
        {
            return output - Vector.SchurProduct(output, output);
        }

        public double InputNeruonSensitivity(int inputNeuronIndex, LearningTask learningTask)
        {
            double totalSensitivity = 0;
            foreach (var learningExample in learningTask)
            {
                totalSensitivity += LearningExampleInputNeuronSensitivity(inputNeuronIndex, learningExample);
            }
            return totalSensitivity / learningTask.Size;
        }

        private double LearningExampleInputNeuronSensitivity(int inputNeuronIndex, LearningExample learningExample)
        {
            ForwardPropagation(learningExample.Input);
            Vector firstHiddenLayerOutput = outputs[1];
            Vector outputDerivative = outputs[1] - Vector.SchurProduct(outputs[1], outputs[1]);
            Vector outputNeuronSensitivities = Vector.SchurProduct(outputDerivative, weights[0][inputNeuronIndex]);
            for (int i = 2; i < NeuronLayerCount; i++)
            {
                Vector derivative = outputs[i] - Vector.SchurProduct(outputs[i], outputs[i]);
                outputNeuronSensitivities = Vector.SchurProduct(derivative, weights[i - 1] * outputNeuronSensitivities);
            }
            return outputNeuronSensitivities.ComponentSum();
        }

        public void PrintSensitivity(LearningTask learningTask)
        {
            Console.WriteLine("Input neuron sensitivities\n-----------------------------------------------");
            for (int i = 0; i < InputDimension; i++)
            {
                Console.WriteLine(InputNeruonSensitivity(i, learningTask));
            }
        }

        /// <summary>
        /// Displays the given inputs and outputs of the given learning examples with the
        /// corresponding neural network output
        /// </summary>
        /// <param name="learningTask">The given learning task to test the perceptron</param>
        public void PrintResults(LearningTask learningTask)
        {
            string dividingLine = "-----------------------------------------------";
            CultureInfo englishCulture = new CultureInfo("en-GB");
            Console.WriteLine("Neural network outputs for given learning task");
            Vector currentNeuralNetworkOutput;
            double totalError = 0;
            foreach (var learningExample in learningTask)
            {
                Console.WriteLine(dividingLine + "\nInput:");
                learningExample.Input.PrintAsRowVector();
                Console.WriteLine("\nOutput:");
                currentNeuralNetworkOutput = ForwardPropagation(learningExample.Input);
                currentNeuralNetworkOutput.PrintAsRowVector();
                Console.WriteLine("\nDesired output:");
                learningExample.Output.PrintAsRowVector();
                // Calculate the error for the given learning task
                totalError += (learningExample.Output - currentNeuralNetworkOutput).SquaredComponentSum();
            }
            Console.WriteLine(dividingLine + "\nTotal error over all learning examples:\n" + String.Format(englishCulture, "{0:F10}", totalError) + "\n");
        }

        /// <summary>
        /// Displays all weights and biases of the neural network in the console
        /// </summary>
        public void PrintNetwork()
        {
            string dividingLine = "-----------------------------------------------";
            Console.WriteLine("Current neural network state\n" + dividingLine);
            if (NeuronLayerCount > 2)
            {

                Console.WriteLine("Weights between input-layer and hidden-layer 1:\n");
                weights[0].Print();
                for (int i = 1; i < weights.Length - 1; i++)
                {
                    Console.WriteLine("\n" + dividingLine + $"\nHidden-layer {i} bias:\n");
                    biases[i - 1].PrintAsRowVector();
                    Console.WriteLine("\n" + dividingLine + $"\nWeights between hidden-layer {i} and hidden-layer {i + 1}:\n");
                    weights[i].Print();
                }
                Console.WriteLine("\n" + dividingLine + $"\nHidden-layer {weights.Length - 1} bias:\n");
                biases[biases.Length - 2].PrintAsRowVector();
                Console.WriteLine("\n" + dividingLine + $"\nWeights between hidden-layer {weights.Length - 1} and output-layer:\n");
                weights[weights.Length - 1].Print();
                Console.WriteLine("\n" + dividingLine + "\nOutput-layer bias:\n");
                biases[biases.Length - 1].PrintAsRowVector();
                Console.WriteLine();
            }
            else
            {
                Console.WriteLine("Weights between input-layer and output-layer:\n");
                weights[0].Print();
                Console.WriteLine("\n" + dividingLine + "\nOutput-layer bias:");
                biases[0].PrintAsRowVector();
                Console.WriteLine();
            }
        }
    }
}
