using System;
using System.Linq;


public class NeuralNetwork
{
    private double[, ] weights;   // Neural network implementation will go here
    enum Operation
    {
        Add,
        Subtract,
        Multiply,
        Divide
    }

    public NeuralNetwork()
    {   
        Random rand = new Random(1);
        int numberofinput = 3;
        int numberofoutput = 1;
        weights = new double[numberofinput, numberofoutput];  //a 3x1 weight matrix

        for (int i = 0; i < numberofinput; i++)
        {
            for (int j = 0; j < numberofoutput; j++)
            {
                weights[i, j] = 2*rand.NextDouble() - 1; //random values between -1 and 1
            }
        }
    }   

    private double[,] Activate(double[, ] matrix, bool isDerivative)
    {
        int row = matrix.GetLength(0);
        int col = matrix.GetLength(1);
        double[, ] result = new double[row, col];

        for(int i = 0; i < row; i++)
        {
            for(int j = 0; j < col; j++)
            {
                if (isDerivative)
                {
                    result[i, j] = matrix[i, j] * (1 - matrix[i, j]);
                }
                else
                {
                    result[i, j] = 1.0 / (1.0 + Math.Exp(-matrix[i, j]));
                }
            }
        }
        return result;
    }

    public void Train(double[, ] TrainingInput , double[, ] TrainingOutput, int iterations) //backpropagation training loop
    {
        for(int iteration = 0; iteration < iterations; iteration++)
        {
            // Forward pass
            double[, ] output = Think(TrainingInput);
            
            // Calculate error
            double[, ] error = PerformOperation(TrainingOutput, output, Operation.Subtract);
            
            // Calculate gradient: error * sigmoid_derivative(output)
            double[, ] gradient = PerformOperation(error, Activate(output, true), Operation.Multiply);
            
            // Calculate weight adjustments: input^T × gradient
            double[, ] adjustments = DotProduct(Transpose(TrainingInput), gradient);
            
            // Update weights
            weights = PerformOperation(weights, adjustments, Operation.Add);
        }
    }

    public double[, ] DotProduct(double[, ] a, double[, ] b)
    {
        int aRows = a.GetLength(0);
        int aCols = a.GetLength(1);
        int bRows = b.GetLength(0);
        int bCols = b.GetLength(1);

        if (aCols != bRows)
            throw new InvalidOperationException("Incompatible matrix dimensions for multiplication.");

        double[, ] result = new double[aRows, bCols];

        for (int i = 0; i < aRows; i++)
        {
            for (int j = 0; j < bCols; j++)
            {
                for (int k = 0; k < aCols; k++)
                {
                    result[i, j] += a[i, k] * b[k, j];
                }
            }
        }

        return result;
    }

    public double[, ] Think(double[, ] input) //forward propagation
    {
        return Activate(DotProduct(input, weights), false);
    }

    private double[, ] PerformOperation(double[, ] a, double[, ] b, Operation operation)
    {
        int row = a.GetLength(0);
        int col = a.GetLength(1);
        double[, ] result = new double[row, col];

        for(int i = 0; i < row; i++)
        {
            for(int j = 0; j < col; j++)
            {
                switch(operation)
                {
                    case Operation.Add:
                        result[i, j] = a[i, j] + b[i, j];
                        break;
                    case Operation.Subtract:
                        result[i, j] = a[i, j] - b[i, j];
                        break;
                    case Operation.Multiply:
                        result[i, j] = a[i, j] * b[i, j];
                        break;
                    case Operation.Divide:
                        result[i, j] = a[i, j] / b[i, j];
                        break;
                }
            }
        }
        return result;
    }

    public double[, ] Transpose(double[, ] matrix)
    {
        return matrix.Transpose(matrix.GetLength(0), matrix.GetLength(1));
    }

     static void PrintMatrix(double[, ] matrix)
    {
        int row = matrix.GetLength(0);
        int col = matrix.GetLength(1);

        for(int i = 0; i < row; i++)
        {
            for(int j = 0; j < col; j++)
            {
                Console.Write($"{Math.Round(matrix[i, j])} ");
            }
            Console.WriteLine();
        }
    }

    static void Main(string[] args)
    {
        NeuralNetwork nn = new NeuralNetwork();
        double[,] trainingInput  = new double[,] { {0, 0, 1}, {1, 1, 1}, {1, 0, 1}, {0, 0, 0} };
        double[,] trainingOutput = new double[,] { {1}, {1}, {1}, {0} };

        nn.Train(trainingInput, trainingOutput, 10000);
        double[,] testInput = new double[,] { {1, 0, 0}, {0, 0, 0}, {0, 1, 0} };
        double[, ] predictions = nn.Think(testInput);

        PrintMatrix(predictions);
    }

   
}

public static class Extentions
{
    public static double[,] Transpose(this double[,] matrix, int rows, int cols)
    {
        double[,] result = new double[cols, rows];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result[j, i] = matrix[i, j];
            }
        }
        return result;
    }
}
