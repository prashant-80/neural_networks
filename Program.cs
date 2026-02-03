using System;


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
        weights = new double[numberofinput, numberofoutput];

        for (int i = 0; i < numberofinput; i++)
        {
            for (int j = 0; j < numberofoutput; j++)
            {
                weights[i, j] = 2*rand.NextDouble() - 1; // Initialize weights to random values between -1 and 1
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
                double sigmoidOutput = result[i, j] =  1/ (1 + Math.Exp(-matrix[i, j]));
                double sigmoidDerivative = result[i, j] = matrix[i, j] * (1 - matrix[i, j]);

                result[i, j] = isDerivative ? sigmoidDerivative : sigmoidOutput;
            }
        }
        return result;
    }

    
}
 