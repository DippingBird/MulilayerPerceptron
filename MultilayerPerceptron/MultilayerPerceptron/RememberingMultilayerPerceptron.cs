using System.Collections.Generic;
using System;

namespace MultilayerPerceptron
{
    /// <summary>
    /// Additionally saves the previous gradient
    /// </summary>
    internal abstract class RememberingMultilayerPerceptron : GeneralMultilayerPerceptron
    {
        protected Matrix[] previousWeightGradients;
        protected Matrix[] weightStepRange;
        protected Vector[] previousBiasGradients;
        protected Vector[] biasStepRange;

        public RememberingMultilayerPerceptron(int inputNeuronLayerDimension, ICollection<int> hiddenNeuronLayerDimensions, int outputNeuronLayerDimension,
        double lowerBound, double higherBound, double weightPunishmentFactor, double startingStepRange)
        : base(inputNeuronLayerDimension, hiddenNeuronLayerDimensions, outputNeuronLayerDimension, lowerBound, higherBound, weightPunishmentFactor)
        {
            Initialize(startingStepRange);
        }

        public RememberingMultilayerPerceptron(Vector[] startingBiases, Matrix[] startingWeights,
        double weightPunishmentFactor, double startingStepRange)
        : base(startingBiases, startingWeights, weightPunishmentFactor)
        {
            Initialize(startingStepRange);
        }

        /// <param name="startingStepRange">The starting value for all weight and bias step-ranges</param>
        private void Initialize(double startingStepRange)
        {
            // Check wether starting step range is positive
            if (startingStepRange <= 0)
            {
                throw new ArgumentException("Starting step range must be positive");
            }
            Function setStartingStepRange = Input => startingStepRange;
            previousWeightGradients = new Matrix[weights.Length];
            weightStepRange = new Matrix[weights.Length];
            previousBiasGradients = new Vector[biases.Length];
            biasStepRange = new Vector[biases.Length];
            for (int i = 0; i < NeuronLayerCount - 1; i++)
            {
                // Initialize previous weight gradients and delta weights
                previousWeightGradients[i] = new Matrix(weights[i].Height, weights[i].Width);
                weightStepRange[i] = new Matrix(weights[i].Height, weights[i].Width);
                weightStepRange[i] = weightStepRange[i].ApplyFunctionOnAllComponents(setStartingStepRange);
                // Initialize previous bias gradients and delta biases
                previousBiasGradients[i] = new Vector(biases[i].Dimension);
                biasStepRange[i] = new Vector(biases[i].Dimension);
                biasStepRange[i] = biasStepRange[i].ApplyFunctionOnAllComponents(setStartingStepRange);
            }
        }

        protected override void AdaptWeightsAndBiases()
        {
            CalculateWeightAndBiasStepRanges();
            ChangeWeightsAndBiases();
        }

        protected abstract void CalculateWeightAndBiasStepRanges();
        private void ChangeWeightsAndBiases()
        {
            for (int i = 0; i < NeuronLayerCount - 1; i++)
            {
                weights[i] += weightStepRange[i];
                biases[i] += biasStepRange[i];
            }
        }

        protected override void ResetWeightAndBiasGradients()
        {
            for (int i = 0; i < NeuronLayerCount - 1; i++)
            {
                previousWeightGradients[i] = currentWeightGradients[i];
                previousBiasGradients[i] = currentBiasGradients[i];
                // Set current gradients to zero
                currentWeightGradients[i] = new Matrix(currentWeightGradients[i].Height, currentWeightGradients[i].Width);
                currentBiasGradients[i] = new Vector(currentBiasGradients[i].Dimension);
            }
        }

        /// <summary>
        /// Changes the sign of the given bias or weight step-range to the gradient sign
        /// </summary>
        /// <param name="currentStepRange">The current step range</param>
        /// <param name="currentGradient">The current gradient</param>
        /// <returns>The signed step range</returns>
        protected double SetStepRangeSign(double currentStepRange, double currentGradient)
        {
            return -Math.Sign(currentGradient) * Math.Abs(currentStepRange);
        }
    }
}
