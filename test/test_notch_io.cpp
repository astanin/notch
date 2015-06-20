#include "catch.hpp"
#include <sstream>
#include <iterator>
using namespace std;


#define NOTCH_ONLY_DECLARATIONS
#include "notch.hpp"
#include "notch_io.hpp"


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

TEST_CASE("Dataset CSV-format reader", "[io]") {
    stringstream csv("0.0,0.0,0.0\n0,1,1\n1,0,1\n1,1,0\n");
    LabeledDataset d = CSVReader<','>::read(csv);
    LabeledDataset expected {{{0,0},{0}}, {{0,1},{1}}, {{1,0},{1}}, {{1,1},{0}}};
    // compare two datasets
    CHECK(d.size() == expected.size());
    CHECK(d.inputDim() == expected.inputDim());
    CHECK(d.outputDim() == expected.outputDim());
    auto d_it = begin(d);
    auto good_it = begin(expected);
    auto good_end = end(expected);
    for (; good_it != good_end; ++d_it, ++good_it) {
        for (size_t i = 0; i < expected.inputDim(); ++i) {
            CHECK((*good_it).data[i] == (*d_it).data[i]);
        }
        for (size_t i = 0; i < expected.outputDim(); ++i) {
            CHECK((*good_it).label[i] == (*d_it).label[i]);
        }
    }
}

TEST_CASE("Dataset CSV-format reader (skiprows)", "[io]") {
    stringstream csv("x,y,XOR\n0.0,0.0,0.0\n0,1,1\n1,0,1\n1,1,0\n");
    LabeledDataset d = CSVReader<','>::read(csv, {2} /* last col */, 1 /* skiprows */);
    LabeledDataset expected {{{0,0},{0}}, {{0,1},{1}}, {{1,0},{1}}, {{1,1},{0}}};
    // compare two datasets
    CHECK(d.size() == expected.size());
    CHECK(d.inputDim() == expected.inputDim());
    CHECK(d.outputDim() == expected.outputDim());
    auto d_it = begin(d);
    auto good_it = begin(expected);
    auto good_end = end(expected);
    for (; good_it != good_end; ++d_it, ++good_it) {
        for (size_t i = 0; i < expected.inputDim(); ++i) {
            CHECK((*good_it).data[i] == (*d_it).data[i]);
        }
        for (size_t i = 0; i < expected.outputDim(); ++i) {
            CHECK((*good_it).label[i] == (*d_it).label[i]);
        }
    }
}

TEST_CASE("Dataset CSV-format reader (categorical labels and quotes)", "[io]") {
    stringstream csv("1,odd\n\"2\",even\n3,odd\n4,\"\"\"four\"\"\"\n");
    LabeledDataset d = CSVReader<>::read(csv);
    LabeledDataset expected {{{1},{0}}, {{2},{1}}, {{3},{0}}, {{4},{2}}};
    // compare two datasets
    CHECK(d.size() == expected.size());
    CHECK(d.inputDim() == expected.inputDim());
    CHECK(d.outputDim() == expected.outputDim());
    auto d_it = begin(d);
    auto good_it = begin(expected);
    auto good_end = end(expected);
    for (; good_it != good_end; ++d_it, ++good_it) {
        for (size_t i = 0; i < expected.inputDim(); ++i) {
            CHECK((*good_it).data[i] == (*d_it).data[i]);
        }
        for (size_t i = 0; i < expected.outputDim(); ++i) {
            CHECK((*good_it).label[i] == (*d_it).label[i]);
        }
    }
}

TEST_CASE("MLP input-output to plain-text", "[io][mlp]") {
    // create a random MLP and write it to string
    stringstream ss;
    Net mlp = MakeNet().MultilayerPerceptron({2, 2, 1}).addL2Loss().init();
    PlainTextNetworkWriter(ss) << mlp;
    // read back
    ss.seekg(0);
    Net mlp2 = PlainTextNetworkReader(ss).read();
    stringstream ss2;
    PlainTextNetworkWriter(ss2) << mlp2;
    // check equivalence
    CHECK(ss.str() == ss2.str());
}
