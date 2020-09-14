using System;
using System.Collections.Generic;

namespace MultilayerPerceptron
{
    class ElasticMultiplayerPerceptron : RememberingMultilayerPerceptron
    {
        double stepRangeShrinkFactor;
        double stepRangeGrowthFactor;
        double minimalStepRange;
        double maximalStepRange;

        public ElasticMultiplayerPerceptron(int inputNeuronLayerDimension, ICollection<int> hiddenNeuronLayerDimensions, int outputNeuronLayerDimension, double lowerBound, double higherBound,
        double weightPunishmentFactor, double startingStepRange, double stepRangeShrinkFactor, double stepRangeGrowthFactor,
        double minimalStepRange, double maximalStepRange) :
        base(inputNeuronLayerDimension, hiddenNeuronLayerDimensions, outputNeuronLayerDimension, lowerBound, higherBound, weightPunishmentFactor, startingStepRange)
        {
            Initialize(stepRangeShrinkFactor, stepRangeGrowthFactor, minimalStepRange, maximalStepRange);
        }

        public ElasticMultiplayerPerceptron(Vector[] startingBiases, Matrix[] startingWeights,
        double weightPunishmentFactor, double startingStepRange, double stepRangeShrinkFactor,
        double stepRangeGrowthFactor, double minimalStepRange, double maximalStepRange) :
        base(startingBiases, startingWeights, weightPunishmentFactor, startingStepRange)
        {
            Initialize(stepRangeShrinkFactor, stepRangeGrowthFactor, minimalStepRange, maximalStepRange);
        }

        /// <param name="stepRangeShrinkFactor">The factor by which the step ranges should be shrinked, commonly
        /// between 0.5 and 0.7</param>
        /// <param name="stepRangeGrowthFactor">The factor by which the step ranges should be growed, commonly
        /// between 1.05 and 1.2</param>
        /// <param name="minimalStepRange">The minimal allowed step range</param>
        /// <param name="maximalStepRange">The maximal allowed step range</param>
        private void Initialize(double stepRangeShrinkFactor, double stepRangeGrowthFactor, double minimalStepRange, double maximalStepRange)
        {
            // Check wether step-range shrink factor is between 0 and 1
            if (stepRangeShrinkFactor <= 0 || stepRangeShrinkFactor > 1)
            {
                throw new ArgumentException("Step-range shrink factor must be between 0 and 1");
            }
            // Check wether step-range growth factor is greater than 1
            if (stepRangeGrowthFactor <= 1)
            {
                throw new ArgumentException("Step-range growth factor must be greater than 1");
            }
            // Check wether minimal and maximal allowed step-ranges must be positive
            if (minimalStepRange < 0 || maximalStepRange <= 0)
            {
                throw new ArgumentException("Minimal and maximal allowed step-ranges must be positive");
            }
            this.stepRangeShrinkFactor = stepRangeShrinkFactor;
            this.stepRangeGrowthFactor = stepRangeGrowthFactor;
            this.minimalStepRange = minimalStepRange;
            this.maximalStepRange = maximalStepRange;
        }

        protected override void CalculateWeightAndBiasStepRanges()
        {
            for (int i = 0; i < NeuronLayerCount - 1; i++)
            {
                // Calculate delta weights
                for (int j = 0; j < weights[i].Height; j++)
                {
                    for (int k = 0; k < weights[i].Width; k++)
                    {
                        weightStepRange[i][j, k] = ChangeStepRange(weightStepRange[i][j, k], currentWeightGradients[i][j, k], previousWeightGradients[i][j, k]);
                        if (currentWeightGradients[i][j, k] * previousWeightGradients[i][j, k] < 0)
                        {
                            currentWeightGradients[i][j, k] = 0;
                        }
                    }
                }
                // Calculate delta biases
                for (int j = 0; j < biases[i].Dimension; j++)
                {
                    biasStepRange[i][j] = ChangeStepRange(biasStepRange[i][j], currentBiasGradients[i][j], previousBiasGradients[i][j]);
                    if (currentBiasGradients[i][j] * previousBiasGradients[i][j] < 0)
                    {
                        currentBiasGradients[i][j] = 0;
                    }
                }
            }
        }


        private double ChangeStepRange(double stepRange, double currentGradient, double oldGradient)
        {
            double gradientProduct = currentGradient * oldGradient;
            if (gradientProduct < 0)
            {
                stepRange *= -stepRangeShrinkFactor;
                if (Math.Abs(stepRange) < minimalStepRange)
                {
                    return SetStepRangeSign(minimalStepRange, currentGradient);
                }
                return stepRange;
            }
            else if (gradientProduct > 0)
            {
                stepRange *= stepRangeGrowthFactor;
                if (Math.Abs(stepRange) > maximalStepRange)
                {
                    return SetStepRangeSign(maximalStepRange, currentGradient);
                }
                return stepRange;
            }
            else
            {
                return SetStepRangeSign(stepRange, currentGradient);
            }
        }
    }
}
