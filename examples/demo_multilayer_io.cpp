#include <iostream>
#include <sstream>

#include "notch.hpp"
#include "notch_io.hpp"

using namespace std;

int main() {
    FullyConnectedLayer layer({1, 2, 3}, {0.5}, linearActivation);
    ostringstream ss;
    Layer_PlainTextFormat fmt(layer);
    ss << fmt;
    cout << "saved:\n---\n" << ss.str() << "---\n";

    FullyConnectedLayer layer_copy(3, 1, defaultTanh);
    istringstream sin(ss.str());
    Layer_PlainTextFormat copy_fmt(layer_copy);
    copy_fmt.load(sin);
    cout << "loaded:\n---\n" << copy_fmt << "---\n";
}
