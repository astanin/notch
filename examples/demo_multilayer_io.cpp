#include <iostream>
#include <sstream>

#include "notch.hpp"
#include "notch_io.hpp"

using namespace std;

int main() {
    stringstream ss;
    FullyConnectedLayer layer({1, 2, 3}, {0.5}, linearActivation);
    PlainTextNetworkWriter(ss) << layer;
    cout << "saved:\n---\n" << ss.str() << "---\n";

    ss.seekg(0);
    FullyConnectedLayer layer_copy;
    PlainTextNetworkReader(ss) >> layer_copy;
    cout << "loaded:\n---\n";
    PlainTextNetworkWriter(cout) << layer_copy;
    cout << "---\n";

}
