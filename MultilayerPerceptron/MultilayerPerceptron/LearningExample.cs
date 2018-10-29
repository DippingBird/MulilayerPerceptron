using System;

namespace MultilayerPerceptron
{
    /// <summary>
    /// A simple learning example for a neural network
    /// </summary>
    class LearningExample
    {
        /// <summary>
        /// Gives the learning example its input and output
        /// </summary>
        /// <param name="input">The neural network input for the given learning example</param>
        /// <param name="output">The desired output of the neural network for the given input</param>
        public LearningExample(Vector input, Vector output)
        {
            Input = input;
            Output = output;
        }

        public Vector Input { get; private set; }
        public Vector Output { get; private set; }

        /// <summary>
        /// Displays the learning-example in the console
        /// </summary>
        public void Print()
        {
            Console.Write("Input:  ");
            Input.PrintAsRowVector();
            Console.Write("Output: ");
            Output.PrintAsRowVector();
        }
    }
}
