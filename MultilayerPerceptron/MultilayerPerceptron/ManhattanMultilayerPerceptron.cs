using System;
using System.Collections.Generic;

namespace MultilayerPerceptron
{
    /// <summary>
    /// Implements a multilayer perceptron with manhattan training
    /// </summary>
    class ManhattanMultilayerPerceptron : GeneralMultilayerPerceptron
    {
        double stepRange;
        Function signFunction;

        public ManhattanMultilayerPerceptron(int inputNeuronLayerDimension, ICollection<int> hiddenNeuronLayerDimensions, int outputNeuronLayerDimension, double lowerBound, double higherBound,
        double weightPunishmentFactor, double stepRange)
        : base(inputNeuronLayerDimension, hiddenNeuronLayerDimensions, outputNeuronLayerDimension, lowerBound, higherBound, weightPunishmentFactor)
        {
            InitializeNetwork(stepRange);
        }

        public ManhattanMultilayerPerceptron(Vector[] startingBiases, Matrix[] startingWeights, double weightPunishmentFactor, double stepRange)
        : base(startingBiases, startingWeights, weightPunishmentFactor)
        {
            InitializeNetwork(stepRange);
        }

        /// <param name="stepRange">The step range for manhattan propagation</param>
        private void InitializeNetwork(double stepRange)
        {
            if (stepRange < 0)
            {
                throw new ArgumentException("Step range must be positive");
            }
            this.stepRange = stepRange;
            signFunction = input => Math.Sign(input);
        }

        /// <summary>
        /// Changes all weights and biases according to the gradients with
        /// manhattan training (constant step range)
        /// </summary>
        protected override void AdaptWeightsAndBiases()
        {
            for (int i = 0; i < NeuronLayerCount - 1; i++)
            {
                // Change weights
                weights[i] -= stepRange * currentWeightGradients[i].ApplyFunctionOnAllComponents(signFunction);
                // Change biases
                biases[i] -= stepRange * currentBiasGradients[i].ApplyFunctionOnAllComponents(signFunction);
            }
        }

        /// <summary>
        /// Sets all weight and bias gradients to zero
        /// </summary>
        protected override void ResetWeightAndBiasGradients()
        {
            for (int i = 0; i < NeuronLayerCount - 1; i++)
            {
                // Set gradients to zero
                currentWeightGradients[i] = new Matrix(currentWeightGradients[i].Height, currentWeightGradients[i].Width);
                currentBiasGradients[i] = new Vector(currentBiasGradients[i].Dimension);
            }
        }
    }
}
