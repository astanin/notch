#ifndef ACTIVATION_H
#define ACTIVATION_H


#include <cmath>      // exp


double sign(double a) { return (a == 0) ? 0 : (a < 0 ? -1 : 1); }


class ActivationFunction {
    public:
        virtual double operator()(double v) const = 0;
        virtual double derivative(double v) const = 0;
};


/// phi(v) = 1/(1 + exp(-slope*v)); Chapter 4, page 135
class LogisticFunction : public ActivationFunction {
    private:
        double slope = 1.0;

    public:
        LogisticFunction(double slope) : slope(slope) {};

        virtual double operator()(double v) const {
            return 1.0/(1.0 + exp(-slope*v));
        }

        virtual double derivative(double v) const {
            double y = (*this)(v);
            return slope * y * (1 - y);
        }
};


/// phi(v) = a * tanh(b * v); Chapter 4, page 136
///
/// Default values for a and b were proposed by (LeCun, 1993),
/// so that phi(1) = 1 and phi(-1) = -1, and the slope at the origin is 1.1424;
/// Chapter 4, page 145.
class TanhFunction : public ActivationFunction {
    private:
        double a;
        double b;
    public:
        TanhFunction(double a=1.7159, double b=0.6667) : a(a), b(b) {}

        virtual double operator()(double v) const {
            return a*tanh(b*v);
        }

        virtual double derivative(double v) const {
            double y = tanh(b*v);
            return a*b*(1.0 - y*y);
        }
};


/// calculate derivative with numeric differentiation
class AutoDiffFunction : public ActivationFunction {
    private:
        function<double(double)> f;
        double dx;
    public:
        AutoDiffFunction(function<double(double)> f, double dx=1e-3) :
            f(f), dx(dx) {}

        virtual double operator()(double v) const {
            return f(v);
        }

        virtual double derivative(double v) const {
            return (f(v+dx) - f(v-dx))/(2.0*dx);
        }
};


const TanhFunction defaultTanh;
const AutoDiffFunction defaultSignum(sign);

#endif /* ACTIVATION_H */
