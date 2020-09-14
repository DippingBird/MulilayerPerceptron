using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;

namespace MultilayerPerceptron
{
    // Function from a real matrix to a real matrix
    delegate Matrix MatrixFunction(Matrix input);

    /// <summary>
    /// Implements a real matrix of variable dimensions
    /// </summary>
    class Matrix : IEnumerable<double>
    {
        double[,] components;
        public Matrix(int height, int width)
        {
            components = new double[height, width];
        }

        /// <summary>
        /// Initializes the matrix with the given components and length
        /// </summary>
        /// <param name="width">The height of the matrix</param>
        /// <param name="components">The elements of the matrix given from the top row left to right to bottom</param>
        public Matrix(int width, params double[] components)
        {
            if (components.Length % width == 0)
            {
                this.components = new double[components.Length / width, width];
                for (int i = 0; i < components.Length; i++)
                {
                    this.components[i / width, i % width] = components[i];
                }
            }
            else
            {
                throw new ArgumentException("Width of matrix does not divide number of elements");
            }
        }

        public Matrix(double[,] components)
        {
            this.components = components;
        }

        public int Width { get => components.GetLength(1); }
        public int Height { get => components.GetLength(0); }
        public int NumberOfElements { get => components.Length; }
        public bool IsQuadratic { get => Width == Height; }

        /// <summary>
        /// Indexer for the individual components of the matrix starting with 0
        /// </summary>
        /// <param name="i">The height index</param>
        /// <param name="j">The width index</param>
        /// <returns>The component identified by the given indices</returns>
        public double this[int i, int j]
        {
            get => components[i, j];
            set => components[i, j] = value;
        }

        /// <summary>
        /// Represents a rowvector of the matrix
        /// </summary>
        /// <param name="i">The rowvector index</param>
        /// <returns>The rowvector from the given index</returns>
        public Vector this[int i]
        {
            get
            {
                Vector rowVector = new Vector(Width);
                for (int j = 0; j < Width; j++)
                {
                    rowVector[j] = components[i, j];
                }
                return rowVector;
            }

            set
            {
                for (int j = 0; j < Width; j++)
                {
                    components[i, j] = value[j];
                }
            }
        }

        /// <summary>
        /// Transposes the matrix
        /// </summary>
        /// <returns>The transposed Matrix</returns>
        public Matrix Transponse()
        {
            Matrix transposedMatrix = new Matrix(Width, Height);
            for (int i = 0; i < Width; i++)
            {
                for (int j = 0; j < Height; j++)
                {
                    transposedMatrix[i, j] = components[j, i];
                }
            }
            return transposedMatrix;
        }

        /// <summary>
        /// Displays the matrix in the console
        /// </summary>
        public void Print()
        {
            string matrix;
            CultureInfo englishCulture = new CultureInfo("en-GB");
            if (Height > 1)
            {
                matrix = "/" + GetRowVectorAsString(englishCulture, 0) + "\\\n";
                for (int i = 1; i < Height - 1; i++)
                {
                    matrix += "|" + GetRowVectorAsString(englishCulture, i) + "|\n";
                }
                matrix += "\\" + GetRowVectorAsString(englishCulture, Height - 1) + "/";
            }
            else
            {
                matrix = "(" + GetRowVectorAsString(englishCulture, 0) + ")";
            }
            Console.WriteLine(matrix);
        }
        private string GetRowVectorAsString(CultureInfo culture, int index)
        {
            string rowVector = "";
            for (int i = 0; i < Width - 1; i++)
            {
                rowVector += String.Format(culture, "{0,7:F3},", components[index, i]);
            }
            rowVector += String.Format(culture, "{0,7:F3}", components[index, Width - 1]);
            return rowVector;
        }

        public IEnumerator<double> GetEnumerator()
        {
            return (IEnumerator<double>)components.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return components.GetEnumerator();
        }

        /// <summary>
        /// Applies the given one dimensional function
        /// on all vector components
        /// </summary>
        /// <param name="function">The one dimensional function to be applied</param>
        /// <returns>The matrix function</returns>
        public Matrix ApplyFunctionOnAllComponents(Function function)
        {
            Matrix transformedMatrix = new Matrix(Height, Width);
            int width = Width;
            int height = Height;
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    transformedMatrix[i, j] = function(components[i, j]);
                }
            }
            return transformedMatrix;
        }

        /// <summary>
        /// Defines the so called Hadamard-product, Schur-product or componentwise-product, 
        /// given matrices must have the same dimensions
        /// </summary>
        /// <param name="v1">first operator matrix</param>
        /// <param name="v2">second operator matrix</param>
        /// <returns>the Schur-product of both given matrices</returns>
        public static Matrix SchurProduct(Matrix matrix1, Matrix matrix2)
        {
            int matrixHeight = matrix1.Height;
            int matrixWidth = matrix1.Width;
            Matrix productMatrix = new Matrix(matrixHeight, matrixWidth);
            for (int i = 0; i < matrixHeight; i++)
            {
                for (int j = 0; j < matrixWidth; j++)
                {
                    productMatrix[i, j] = matrix1[i, j] * matrix2[i, j];
                }
            }
            return productMatrix;
        }

        public static Matrix operator *(Matrix matrix1, Matrix matrix2)
        {
            int matrix1Height = matrix1.Height;
            int matrix2Width = matrix2.Width;
            int commonDimension = matrix1.Width;
            Matrix matrixProduct = new Matrix(matrix1Height, matrix2Width);
            double currentScalarProduct;
            for (int i = 0; i < matrix1Height; i++)
            {
                for (int j = 0; j < matrix2Width; j++)
                {
                    currentScalarProduct = 0;
                    for (int k = 0; k < commonDimension; k++)
                    {
                        currentScalarProduct += matrix1[i, k] * matrix2[k, j];
                    }
                    matrixProduct[i, j] = currentScalarProduct;
                }
            }
            return matrixProduct;
        }

        public static Vector operator *(Matrix matrix, Vector vector)
        {
            int vectorDimension = vector.Dimension;
            int matrixHeight = matrix.Height;
            Vector transformedVector = new Vector(matrixHeight);
            double currentScalarProduct;
            // For all rowvectors
            for (int i = 0; i < matrixHeight; i++)
            {
                currentScalarProduct = 0;
                // Scalarproduct between the rowvector of the matrix and the input-vector
                for (int j = 0; j < vectorDimension; j++)
                {
                    currentScalarProduct += matrix[i, j] * vector[j];
                }
                transformedVector[i] = currentScalarProduct;
            }
            return transformedVector;
        }

        public static Matrix operator *(double scalar, Matrix matrix)
        {
            int matrixHeight = matrix.Height;
            int matrixWidth = matrix.Width;
            Matrix scaledMatrix = new Matrix(matrixHeight, matrixWidth);
            // Multiply every matrix element with the scalar
            for (int i = 0; i < matrixHeight; i++)
            {
                for (int j = 0; j < matrixWidth; j++)
                {
                    scaledMatrix[i, j] = matrix[i, j] * scalar;
                }
            }
            return scaledMatrix;
        }

        public static Matrix operator *(Matrix matrix, double scalar)
        {
            return scalar * matrix;
        }

        public static Matrix operator +(Matrix matrix1, Matrix matrix2)
        {
            int matrixHeight = matrix1.Height;
            int matrixWidth = matrix1.Width;
            Matrix matrixSum = new Matrix(matrixHeight, matrixWidth);
            for (int i = 0; i < matrixHeight; i++)
            {
                for (int j = 0; j < matrixWidth; j++)
                {
                    matrixSum[i, j] = matrix1[i, j] + matrix2[i, j];
                }
            }
            return matrixSum;
        }

        public static Matrix operator -(Matrix matrix1, Matrix matrix2)
        {
            int matrixHeight = matrix1.Height;
            int matrixWidth = matrix1.Width;
            Matrix matrixDifference = new Matrix(matrixHeight, matrixWidth);
            for (int i = 0; i < matrixHeight; i++)
            {
                for (int j = 0; j < matrixWidth; j++)
                {
                    matrixDifference[i, j] = matrix1[i, j] - matrix2[i, j];
                }
            }
            return matrixDifference;
        }
    }
}
