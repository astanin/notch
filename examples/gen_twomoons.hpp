#ifndef GEN_TWOMOONS_H
#define GEN_TWOMOONS_H

/// gen_twomoons.hpp --generate training sets for two-moons classification problem
///
/// See NNLM3, Chapter 1. Section 1.5.

#include <memory>
#include <random>
#include "notch.hpp"


const float default_r = 10.0; // radius
const float default_w = 6.0;  // width
const float default_d = 1.0;  // distance (separation) between moons
const float default_n = 1000; // the number of points


LabeledDataset generate(float r=default_r, float w=default_w,
                        float d=default_d, int n=default_n) {
    LabeledDataset data;
    float epsilon = 1e-6 * w;
    std::uniform_real_distribution<> x1(-r - 0.5 * w, r + 0.5 * w + epsilon);
    std::uniform_real_distribution<> y1(0.0, r + 0.5 * w + epsilon);
    std::uniform_real_distribution<> x2(-0.5 * w, 2 * r + 0.5 * w + epsilon);
    std::uniform_real_distribution<> y2(-d - r - 0.5 * w, -d + epsilon);
    std::unique_ptr<RNG> rng = newRNG();
    auto inMoon1 = [r, w](float x, float y) {
        float rr = sqrt(x * x + y * y);
        return rr >= (r - 0.5 * w) && rr <= (r + 0.5 * w);
    };
    auto inMoon2 = [r, w, d](float x, float y) {
        float x_ = x - r;
        float y_ = y - (-d);
        float rr = sqrt(x_ * x_ + y_ * y_);
        return rr >= (r - 0.5 * w) && rr <= (r + 0.5 * w);
    };
    int n1 = 0;
    int n2 = 0;
    while (n1 < n / 2) {
        float x = x1(*rng);
        float y = y1(*rng);
        if (inMoon1(x, y)) {
            Input input = {x, y};
            Output output = {+1};
            data.append(input, output);
            n1++;
        }
    }
    while ((n1 + n2) < n) {
        float x = x2(*rng);
        float y = y2(*rng);
        if (inMoon2(x, y)) {
            Input input = {x, y};
            Output output = {-1};
            data.append(input, output);
            n2++;
        }
    }
    return data;
}

#endif /* GEN_TWOMOONS_H */
