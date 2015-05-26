#ifndef TEST_NOTCH_IO_HPP
#define TEST_NOTCH_IO_HPP

#include <sstream>

#include "catch.hpp"
#include "notch_io.hpp"


using namespace std;


TEST_CASE("Array plain-text I/O", "[io]") {
    Array a = {0.5, 1.0, 2.0, 4.0};
    // output
    ostringstream sout;
    sout << a;
    CHECK(sout.str() == "0.5 1 2 4");
    // input
    Array b(4);
    istringstream sin(sout.str());
    sin >> b;
    REQUIRE(b.size() == a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        CHECK(b[i] == a[i]);
    }
}

#endif
