using ML;
using static System.Console;

double[][] input =
{
    new double[] { 0, 0 },
    new double[] { 0, 1 },
    new double[] { 1, 0 },
    new double[] { 1, 1 }
};
double[][] output =
{
    new double[] { 0 },
    new double[] { 1 },
    new double[] { 1 },
    new double[] { 0 }
};

Network network = new(2, 1);
network.Init(input, output);
//WriteLine(network);

WriteLine("\nEstimates:" + network.Predict());
WriteLine("\nError:"     + network.Error());
// WriteLine(network);