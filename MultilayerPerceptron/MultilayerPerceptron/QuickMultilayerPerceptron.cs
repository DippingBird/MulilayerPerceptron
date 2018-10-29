using System;
using System.Collections.Generic;

namespace MultilayerPerceptron
{
    /// <summary>
    /// Implemets a multilayer-perceptron with quick propagation and weight-decay, and modified gradient descent
    /// </summary>
    class QuickMultilayerPerceptron : RememberingMultilayerPerceptron
    {
        double maximalStepRange;
        double learnRate;

        public QuickMultilayerPerceptron(int inputNeuronLayerDimension, ICollection<int> hiddenNeuronLayerDimensions, int outputNeuronLayerDimension, double lowerBound, double higherBound,
        double weightPunishmentFactor, double startingStepRange, double maximalStepRange, double learnRate) :
        base(inputNeuronLayerDimension, hiddenNeuronLayerDimensions, outputNeuronLayerDimension, lowerBound, higherBound, weightPunishmentFactor, startingStepRange)
        {
            Initialize(maximalStepRange, learnRate);
        }

        public QuickMultilayerPerceptron(Vector[] startingBiases, Matrix[] startingWeights,
        double weightPunishmentFactor, double startingStepRange, double maximalStepRange, double learnRate) :
        base(startingBiases, startingWeights, weightPunishmentFactor, startingStepRange)
        {
            Initialize(maximalStepRange, learnRate);
        }

        /// <param name="maximalStepRange">Maximal allowed step range</param>
        /// <param name="learnRate">Learn rate if a "jump" is too large or the parabola isnt opened to the top</param>
        private void Initialize(double maximalStepRange, double learnRate)
        {
            // Check wether maximal allowed step range is positive
            if (maximalStepRange < 0)
            {
                throw new ArgumentException("Maximal allowed step range must be positive");
            }
            if (learnRate < 0 || learnRate > 1)
            {
                throw new ArgumentException("Manhattan learn rate must be between 0 and 1");
            }
            this.maximalStepRange = maximalStepRange;
            this.learnRate = learnRate;
        }

        /// <summary>
        /// Calculates the step-ranges for the biases and weights according to quick-propagation
        /// </summary>
        protected override void CalculateWeightAndBiasStepRanges()
        {
            for (int i = 0; i < NeuronLayerCount - 1; i++)
            {
                // Calculate weight step ranges
                for (int j = 0; j < weights[i].Height; j++)
                {
                    for (int k = 0; k < weights[i].Width; k++)
                    {
                        weightStepRange[i][j, k] = NextStepRange(weightStepRange[i][j, k], currentWeightGradients[i][j, k], previousWeightGradients[i][j, k]);
                    }
                }
                // Calculate bias step ranges
                for (int j = 0; j < biases[i].Dimension; j++)
                {
                    biasStepRange[i][j] = NextStepRange(biasStepRange[i][j], currentBiasGradients[i][j], previousBiasGradients[i][j]);
                }
            }
        }

        /// <summary>
        /// Approximates the error function of a bias or weight localy with a parabola
        /// </summary>
        /// <param name="currentStepRange">The current step range of a bias or weight</param>
        /// <param name="currentGradient">The gradient of the current iteration of a bias or weight</param>
        /// <param name="previousGradient">The gradient of the previous iteration of a bias or weight</param>
        /// <returns></returns>
        private double NextStepRange(double currentStepRange, double currentGradient, double previousGradient)
        {
            double gradientProduct = currentGradient * previousGradient;
            // If the previous and the old gradient show in the same direction            
            if (gradientProduct > 0)
            {
                // Use manhattan descend
                return currentStepRange;
            }
            else if (gradientProduct < 0)
            {
                double controlFactor = (previousGradient - currentGradient) / currentStepRange;
                // Check wether the parabola is opened to the top
                if (controlFactor < 0)
                {
                    double nextStepRange = currentStepRange * currentGradient / controlFactor;
                    // If the next step range doesnt exeed the maximal allowed step range
                    if (Math.Abs(nextStepRange) < maximalStepRange)
                    {
                        // Move to the minimum of the parabola
                        return nextStepRange;
                    }
                }
                // Otherwise normal gradient-decsend
                return -learnRate * currentStepRange;
            }
            else
            {
                return SetStepRangeSign(currentStepRange, currentGradient);
            }
        }
    }

}
