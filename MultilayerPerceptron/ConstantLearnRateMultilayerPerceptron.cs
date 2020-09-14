using System;
using System.Collections.Generic;

namespace MultilayerPerceptron
{
    /// <summary>
    /// Implements flat spot elemination and a constant learn rate
    /// </summary>
    internal abstract class ConstantLearnRateMultilayerPerceptron : GeneralMultilayerPerceptron
    {
        protected double learnRate;
        private double flatSpotEleminationCoefficient;

        public ConstantLearnRateMultilayerPerceptron(int inputNeuronLayerDimension, ICollection<int> hiddenNeuronLayerDimensions,
        int outputNeuronLayerDimension, double lowerBound, double higherBound,
        double weightPunishmentFactor, double learnRate, double flatSpotEleminationCoefficient)
        : base(inputNeuronLayerDimension, hiddenNeuronLayerDimensions, outputNeuronLayerDimension, lowerBound, higherBound, weightPunishmentFactor)
        {
            Initialize(learnRate, flatSpotEleminationCoefficient);
        }

        public ConstantLearnRateMultilayerPerceptron(Vector[] startingBiases, Matrix[] startingWeights,
        double weightPunishmentFactor, double learnRate, double flatSpotEleminationCoefficient)
        : base(startingBiases, startingWeights, weightPunishmentFactor)
        {
            Initialize(learnRate, flatSpotEleminationCoefficient);
        }

        /// <param name="learnRate">The learn rate for the gradient</param>
        /// <param name="flatSpotEleminationCoefficient">The value by which the logistic derivative should be increased, commonly 0.1</param>
        private void Initialize(double learnRate, double flatSpotEleminationCoefficient)
        {
            // Check wether learn rate is positive
            if (learnRate <= 0)
            {
                throw new ArgumentException("Learn rate must be positive");
            }
            // Check wether flat-spot elemination coefficient is positive or zero
            if (flatSpotEleminationCoefficient < 0)
            {
                throw new ArgumentException("Flat-spot elemination coefficient must be positive or equal to zero");
            }
            this.learnRate = learnRate;
            this.flatSpotEleminationCoefficient = flatSpotEleminationCoefficient;
        }

        /// <summary>
        /// Changes all weights and biases according to the gradients
        /// </summary>
        protected override void AdaptWeightsAndBiases()
        {
            for (int i = 0; i < NeuronLayerCount - 1; i++)
            {
                // Change weights
                weights[i] -= learnRate * currentWeightGradients[i];
                // Change biases
                biases[i] -= learnRate * currentBiasGradients[i];
            }
        }

        /// <summary>
        /// Increases the derivative artificially in order to increase
        /// learning speed on flat planes on the error function
        /// </summary>
        /// <param name="output">The modified derivative</param>
        /// <returns></returns>
        protected override Vector OutputDerivative(Vector output)
        {
            return (base.OutputDerivative(output)).AddToAllComponents(flatSpotEleminationCoefficient);
        }
    }
}
