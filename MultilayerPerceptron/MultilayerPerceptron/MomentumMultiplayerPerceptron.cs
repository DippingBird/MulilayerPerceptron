using System;
using System.Collections.Generic;

namespace MultilayerPerceptron
{
    /// <summary>
    /// Implements a multilayer perceptron with momentum term training, flat spot elemination
    /// and weight punishment
    /// </summary>
    class MomentumMultiplayerPerceptron : ConstantLearnRateMultilayerPerceptron
    {
        double momentumTermCoefficient;

        public MomentumMultiplayerPerceptron(Vector[] startingBiases, Matrix[] startingWeights, double weightPunishmentFactor,
        double learnRate, double flatSpotEleminationCoefficient, double momentumTermCoefficient)
        : base(startingBiases, startingWeights, weightPunishmentFactor, learnRate, flatSpotEleminationCoefficient)
        {
            Initialize(momentumTermCoefficient);
        }

        public MomentumMultiplayerPerceptron(int inputNeuronLayerDimension, ICollection<int> hiddenNeuronLayerDimensions, int outputNeuronLayerDimension,
        double lowerBound, double higherBound, double weightPunishmentFactor, double learnRate, double flatSpotEleminationCoefficient,
        double momentumTermCoefficient)
        : base(inputNeuronLayerDimension, hiddenNeuronLayerDimensions, outputNeuronLayerDimension, lowerBound, higherBound, weightPunishmentFactor,
        learnRate, flatSpotEleminationCoefficient)
        {
            Initialize(momentumTermCoefficient);
        }

        /// <param name="momentumTermCoefficient">The coefficient by which the previous step ranges should be preserved,
        /// commonly between 0.5 and 0.95</param>
        private void Initialize(double momentumTermCoefficient)
        {
            // Check wether momentum term coefficient is between 0 and 1
            if (momentumTermCoefficient < 0 || momentumTermCoefficient > 1)
            {
                throw new ArgumentException("Momentum term coefficient must be between 0 and 1");
            }
            this.momentumTermCoefficient = momentumTermCoefficient;
        }

        /// <summary>
        /// Resets the gradient according to momentum term training
        /// </summary>
        protected override void ResetWeightAndBiasGradients()
        {
            for (int i = 0; i < NeuronLayerCount - 1; i++)
            {
                // Multiply gradients by momentum term factor
                currentWeightGradients[i] *= momentumTermCoefficient;
                currentBiasGradients[i] *= momentumTermCoefficient;
            }
        }
    }
}
