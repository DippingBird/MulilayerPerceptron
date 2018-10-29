using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;

namespace MultilayerPerceptron
{
    // Simple one dimensional real function
    delegate double Function(double input);

    /// <summary>
    /// Implements a real vector of variable dimension
    /// </summary>
    class Vector : IEnumerable<double>
    {
        double[] components;

        public Vector(int dimension)
        {
            components = new double[dimension];
        }

        public Vector(params double[] components)
        {
            this.components = components;
        }

        public int Dimension { get => components.Length; }

        /// <summary>
        /// The euclidian length of the vector
        /// </summary>
        public double Length { get => Math.Sqrt(SquaredComponentSum()); }

        /// <summary>
        /// Indexer for the individual components of the vector starting with 0
        /// </summary>
        /// <param name="index">The index identifying the componen</param>
        /// <returns>The component identified by the index</returns>
        public double this[int index]
        {
            get => components[index];
            set => components[index] = value;
        }

        /// <summary>
        /// Adds all components of the vector together
        /// </summary>
        /// <returns>The sum of all components</returns>
        public double ComponentSum()
        {
            double componentSum = 0;
            for (int i = 0; i < components.Length; i++)
            {
                componentSum += components[i];
            }
            return componentSum;
        }

        /// <summary>
        /// Adds all squared components of the vector together
        /// </summary>
        /// <returns>The squared sum of all components</returns>
        public double SquaredComponentSum()
        {
            double squaredComponentSum = 0;
            for (int i = 0; i < components.Length; i++)
            {
                squaredComponentSum += components[i] * components[i];
            }
            return squaredComponentSum;
        }

        /// <summary>
        /// Adds the given real number to all vector components
        /// </summary>
        /// <param name="input">The given real value to add</param>
        /// <returns>The sum of the given real value and the vector</returns>
        public Vector AddToAllComponents(double input)
        {
            Vector sumVector = new Vector(components.Length);
            for (int i = 0; i < components.Length; i++)
            {
                sumVector[i] = components[i] + input;
            }
            return sumVector;
        }

        /// <summary>
        /// Displays the vector as a row-vector in the console
        /// </summary>
        public void PrintAsRowVector()
        {
            string vector = "(";
            CultureInfo englishCulture = new CultureInfo("en-GB");
            for (int i = 0; i < components.Length - 1; i++)
            {
                vector += String.Format(englishCulture, "{0,7:F3},", components[i]);

            }
            vector += String.Format(englishCulture, "{0,7:F3})", components[components.Length - 1]);
            Console.WriteLine(vector);
        }

        /// <summary>
        /// Displays the vector as a column-vector in the console
        /// </summary>
        public void PrintAsColumnVector()
        {
            string vector = "";
            CultureInfo englishCulture = new CultureInfo("en-GB");
            for (int i = 0; i < components.Length; i++)
            {
                vector += String.Format(englishCulture, "({0,7:F3})\n", components[i]);

            }
            Console.Write(vector);
        }

        /// <summary>
        /// Applies the given one dimensional function
        /// on all vector components
        /// </summary>
        /// <param name="function">The one dimensional function to be applied</param>
        /// <returns>The vector function</returns>
        public Vector ApplyFunctionOnAllComponents(Function function)
        {
            Vector transformedVector = new Vector(Dimension);
            for (int i = 0; i < Dimension; i++)
            {
                transformedVector[i] = function(components[i]);
            }
            return transformedVector;
        }

        /// <summary>
        /// Defines the so called Hadamard-product, Schur-product or componentwise-product, 
        /// given vectors must have the same length
        /// </summary>
        /// <param name="v1">first operator vector</param>
        /// <param name="v2">second operator vector</param>
        /// <returns>the Schur-product of both given vectors</returns>
        public static Vector SchurProduct(Vector v1, Vector v2)
        {
            Vector vectorProduct = new Vector(v1.Dimension);
            for (int i = 0; i < v1.Dimension; i++)
            {
                vectorProduct[i] = v1[i] * v2[i];
            }
            return vectorProduct;
        }

        public IEnumerator<double> GetEnumerator()
        {
            return (IEnumerator<double>)components.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return components.GetEnumerator();
        }

        public static double operator *(Vector vector1, Vector vector2)
        {
            double scalarProduct = 0;
            for (int i = 0; i < vector1.Dimension; i++)
            {
                scalarProduct *= vector1[i] * vector2[i];
            }
            return scalarProduct;
        }

        public static Vector operator +(Vector vector1, Vector vector2)
        {
            Vector vectorSum = new Vector(vector1.Dimension);
            for (int i = 0; i < vector1.Dimension; i++)
            {
                vectorSum[i] = vector1[i] + vector2[i];
            }
            return vectorSum;
        }

        public static Vector operator -(Vector vector1, Vector vector2)
        {
            Vector vectorSum = new Vector(vector1.Dimension);
            for (int i = 0; i < vector1.Dimension; i++)
            {
                vectorSum[i] = vector1[i] - vector2[i];
            }
            return vectorSum;
        }

        public static Vector operator *(double scalar, Vector vector)
        {
            Vector scaledVector = new Vector(vector.Dimension);
            for (int i = 0; i < vector.Dimension; i++)
            {
                scaledVector[i] = scalar * vector[i];
            }
            return scaledVector;
        }

        public static Vector operator *(Vector vector, double scalar)
        {
            return scalar * vector;
        }
    }
}
