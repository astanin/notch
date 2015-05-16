#ifndef ACTIVATION_H
#define ACTIVATION_H


#include <functional> // function<>
#include <cmath>      // exp


double sign(double a) { return (a == 0) ? 0 : (a < 0 ? -1 : 1); }


class ActivationFunction {
public:
    virtual double operator()(double v) const = 0;
    virtual double derivative(double v) const = 0;
    virtual void print(std::ostream &out) const = 0;
};


std::ostream &operator<<(std::ostream &out, const ActivationFunction &af) {
    af.print(out);
    return out;
}


/// phi(v) = 1/(1 + exp(-slope*v)); Chapter 4, page 135
class LogisticFunction : public ActivationFunction {
private:
    double slope = 1.0;

public:
    LogisticFunction(double slope) : slope(slope){};

    virtual double operator()(double v) const {
        return 1.0 / (1.0 + exp(-slope * v));
    }

    virtual double derivative(double v) const {
        double y = (*this)(v);
        return slope * y * (1 - y);
    }

    virtual void print(std::ostream &out) const { out << "logistic"; }
};


class SignumFunction : public ActivationFunction {
public:
    SignumFunction() {}

    virtual double operator()(double v) const { return sign(v); }

    virtual double derivative(double) const { return 0.0; }

    virtual void print(std::ostream &out) const { out << "sign"; }
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
    TanhFunction(double a = 1.7159, double b = 0.6667) : a(a), b(b) {}

    virtual double operator()(double v) const { return a * tanh(b * v); }

    virtual double derivative(double v) const {
        double y = tanh(b * v);
        return a * b * (1.0 - y * y);
    }

    virtual void print(std::ostream &out) const { out << "tanh"; }
};


class PiecewiseLinearFunction : public ActivationFunction {
private:
    double negativeSlope;
    double positiveSlope;
    std::string name;

public:
    PiecewiseLinearFunction(double negativeSlope = 0.0,
                            double positiveSlope = 1.0,
                            std::string name = "ReLU")
        : negativeSlope(negativeSlope), positiveSlope(positiveSlope), name(name) {}

    virtual double operator()(double v) const {
        if (v >= 0) {
            return positiveSlope * v;
        } else {
            return negativeSlope * v;
        }
    }

    virtual double derivative(double v) const {
        if (v >= 0) {
            return positiveSlope;
        } else {
            return negativeSlope;
        }
    }

    virtual void print(std::ostream &out) const { out << name; }
};


const TanhFunction defaultTanh(1.0, 1.0);
const TanhFunction scaledTanh;
const SignumFunction defaultSignum;
const PiecewiseLinearFunction ReLU;
const PiecewiseLinearFunction leakyReLU(0.01, 1.0, "leakyReLU");
const PiecewiseLinearFunction linearActivation(1.0, 1.0, "");


#endif /* ACTIVATION_H */
