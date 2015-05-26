#include <iostream>
#include <sstream>

#include "notch.hpp"
#include "notch_io.hpp"

using namespace std;

int main() {
    FullyConnectedLayer layer({1, 2, 3}, {0.5}, linearActivation);
    stringstream ss;
    ss << PlainTextFormatLayer(layer);
    cout << "saved:\n---\n" << ss.str() << "---\n";
}
