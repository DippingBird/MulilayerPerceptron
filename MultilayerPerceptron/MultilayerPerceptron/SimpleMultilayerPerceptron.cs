using System.Collections.Generic;

namespace MultilayerPerceptron
{
    /// <summary>
    /// Implements a simple neural network with a learn rate and no special gradient decsend
    /// </summary>
    class SimpleMultilayerPerceptron : ConstantLearnRateMultilayerPerceptron
    {
        public SimpleMultilayerPerceptron(int inputNeuronLayerDimension, ICollection<int> hiddenNeuronLayerDimensions, int outputNeuronLayerDimension,
        double lowerBound, double higherBound, double weightPunishmentFactor, double learnRate, double flatSpotEleminationCoefficient)
        : base(inputNeuronLayerDimension, hiddenNeuronLayerDimensions, outputNeuronLayerDimension, lowerBound,
        higherBound, weightPunishmentFactor, learnRate, flatSpotEleminationCoefficient)
        {
        }

        public SimpleMultilayerPerceptron(Vector[] startingBiases, Matrix[] startingWeights,
        double weightPunishmentFactor, double learnRate, double flatSpotEleminationCoefficient) :
        base(startingBiases, startingWeights, weightPunishmentFactor, learnRate, flatSpotEleminationCoefficient)
        {
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
