/// Generate training data sets for float-moon classification problem
/// Chapter 1. Section 1.5.

#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include <algorithm> // any_of


#include "notch.hpp"
#include "notch_io.hpp"
#include "gen_twomoons.hpp"


using namespace std;


void usage() {
    cout << "usage: gen_twomoons distance numpoints output_file";
    exit(-1);
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
        auto dataset = generate(default_r, default_w, d, n);
        out << FANNFormat(dataset);
    } else {
        usage();
    }
}
