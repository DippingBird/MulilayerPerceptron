using System;
using System.Collections.Generic;

namespace MultilayerPerceptron
{
    class Program
    {
        static void Main(string[] args)
        {
            LearningTask trainXOR = new LearningTask(2, 1);
            trainXOR.Add(new Vector(0, 0), new Vector(0.0));
            trainXOR.Add(new Vector(0, 1), new Vector(1.0));
            trainXOR.Add(new Vector(1, 0), new Vector(1.0));
            trainXOR.Add(new Vector(1, 1), new Vector(1.0));
            int[] hiddenNeuronLayers = new int[] { 2 };
            var testNeuralNetwork = new SimpleMultilayerPerceptron(2, hiddenNeuronLayers, 1, -1, 1, 0, 1, 0.1);
            //var testNeuralNetwork = new ManhattanMultilayerPerceptron(2, hiddenNeuronLayers, 1, -1, 1, 0, 0.01);
            //var testNeuralNetwork = new MomentumMultiplayerPerceptron(2, hiddenNeuronLayers, 1, -1, 1, 0.0005, 0.1, 0.1, 0.95);
            //var testNeuralNetwork = new ElasticMultiplayerPerceptron(2, hiddenNeuronLayers, 1, -1, 1, 0.01, 0.1, 0.6, 1.1, 0.05, 0.2);
            //var testNeuralNetwork = new QuickMultilayerPerceptron(2, hiddenNeuronLayers, 1, -1, 1, 0, 0.1, 1, 0.3);
            for (int i = 0; i < 100; i++)
            {
                testNeuralNetwork.BatchLearning(trainXOR);
            }
            testNeuralNetwork.PrintResults(trainXOR);
            testNeuralNetwork.PrintNetwork();
            testNeuralNetwork.PrintSensitivity(trainXOR);
        }
    }
}
