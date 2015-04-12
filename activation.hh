#ifndef ACTIVATION_H
#define ACTIVATION_H


#include <cmath>      // exp


double sign(double a) { return (a == 0) ? 0 : (a < 0 ? -1 : 1); }


class ActivationFunction {
    public:
        virtual double operator()(double v) const = 0;
        virtual double derivative(double v) const = 0;
};


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


AutoDiffFunction signumFunction(sign);

#endif /* ACTIVATION_H */
