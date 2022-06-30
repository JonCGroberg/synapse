namespace ML;

/// <summary>
///     Class <c>NormalDist</c> is a BoxMullerTransform: Creates two uniform distributions; Returns a normal distribution
/// </summary>
public class NormalDist : Random
{
    public double NextDouble(double μ, double σ)
    {
        double μ1                 = 1.0 - base.NextDouble();
        double μ2                 = 1.0 - base.NextDouble();
        double standardNormalDist = Math.Sqrt(-2.0 * Math.Log(μ1)) * Math.Sin(2.0 * Math.PI * μ2);
        double normalDistribution = μ + σ * standardNormalDist;
        return normalDistribution;
    }

    public override double NextDouble()
    {
        const double μ = 0;
        const double σ = 1;
        return NextDouble(μ, σ);
    }
}