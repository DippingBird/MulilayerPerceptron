using System;
using System.Collections;
using System.Collections.Generic;

namespace MultilayerPerceptron
{
    /// <summary>
    /// Implements a simple learning-task consisting of multiple learning-examples, 
    /// all with the same input- and output-dimension
    /// </summary>
    class LearningTask : IEnumerable<LearningExample>
    {
        List<LearningExample> learningTask;
        /// <summary>
        /// Initializes the learning task with given dimensions
        /// </summary>
        /// <param name="inputDimension">The input dimesion of the learning-examples for the task</param>
        /// <param name="outputDimesion">The output dimesion of the learning-examples for the task</param>
        public LearningTask(int inputDimension, int outputDimesion)
        {
            if (inputDimension < 1 || outputDimesion < 1)
            {
                throw new ArgumentException("Input- and outputdimension must be greater or equal to 1");
            }
            learningTask = new List<LearningExample>();
            InputDimension = inputDimension;
            OutputDimension = outputDimesion;
        }
        public int InputDimension { get; private set; }
        public int OutputDimension { get; private set; }
        public int Size { get => learningTask.Count; }

        /// <summary>
        /// Adds the given input and output as learning-example to the task
        /// </summary>
        /// <param name="input">The input of the learning-example</param>
        /// <param name="output">The output of the learning-example</param>
        public void Add(Vector input, Vector output)
        {
            Add(new LearningExample(input, output));
        }

        /// <summary>
        /// Adds the given learning-example to the task
        /// </summary>
        /// <param name="learningExample">The learning-example to add</param>
        public void Add(LearningExample learningExample)
        {
            // Check wether learning.example has the same dimesions as the task
            if (learningExample.Input.Dimension == InputDimension && learningExample.Output.Dimension == OutputDimension)
            {
                learningTask.Add(learningExample);
            }
            else
            {
                throw new ArgumentException("Dimensions of given learning-example do not match learning-task");
            }
        }

        /// <summary>
        /// Displays all learning-examples of the learning-task in the console
        /// </summary>
        public void Print()
        {
            string dividingLine = "-----------------------------------------------";
            Console.WriteLine("Learning examples of learning task\n" + dividingLine);
            Console.WriteLine($"Input dimension: {InputDimension}, Output Dimension: {OutputDimension}");
            foreach (var example in learningTask)
            {
                Console.WriteLine(dividingLine);
                example.Print();
            }
        }

        public IEnumerator<LearningExample> GetEnumerator()
        {
            return learningTask.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return learningTask.GetEnumerator();
        }
    }
}
