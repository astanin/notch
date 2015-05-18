/// Generate training data sets for float-moon classification problem
/// Chapter 1. Section 1.5.

#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include <algorithm> // any_of


#include "notch.hpp"
#include "notch_io.hpp"


using namespace std;


const float r = 10.0; // radius
const float w = 6.0;  // width
const float default_d = 1.0;
const float default_n = 1000;


void usage() {
    cout << "usage: gen_twomoons distance [numpoints [output_file]]";
    cout << "\ndefault options:\n";
    cout << "    distance = 1.0\n";
    cout << "    numpoints = 1000\n";
    cout << "    output_file = stdout\n";
    exit(-1);
}


void generate(float r, float w, float d, int n, ostream &out) {
    LabeledDataset data;
    float epsilon = 1e-6 * w;
    uniform_real_distribution<> x1(-r - 0.5 * w, r + 0.5 * w + epsilon);
    uniform_real_distribution<> y1(0.0, r + 0.5 * w + epsilon);
    uniform_real_distribution<> x2(-0.5 * w, 2 * r + 0.5 * w + epsilon);
    uniform_real_distribution<> y2(-d - r - 0.5 * w, -d + epsilon);
    random_device rd;
    mt19937 rng(rd());
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
        float x = x1(rng);
        float y = y1(rng);
        if (inMoon1(x, y)) {
            Input input = {x, y};
            Output output = {+1};
            data.append(input, output);
            n1++;
        }
    }
    while ((n1 + n2) < n) {
        float x = x2(rng);
        float y = y2(rng);
        if (inMoon2(x, y)) {
            Input input = {x, y};
            Output output = {-1};
            data.append(input, output);
            n2++;
        }
    }
    out << data;
}


int main(int argc, char *argv[]) {
    auto is_help_flag = [](const char *arg) {
        return string(arg) == "--help" || string(arg) == "-h";
    };
    if ((argc > 1) && any_of(argv + 1, argv + argc, is_help_flag)) {
        usage();
    } else if (argc == 1 + 3) {
        float d = atof(argv[1]);
        int n = atoi(argv[2]);
        ofstream out(argv[3], ios::out);
        generate(r, w, d, n, out);
    } else if (argc == 1 + 2) {
        float d = atof(argv[1]);
        int n = atoi(argv[2]);
        generate(r, w, d, n, cout);
    } else if (argc == 1 + 1) {
        float d = atof(argv[1]);
        int n = default_n;
        generate(r, w, d, n, cout);
    } else {
        usage();
    }
}
