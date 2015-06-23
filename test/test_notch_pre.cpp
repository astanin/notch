#include "catch.hpp"

#define NOTCH_ONLY_DECLARATIONS
#include "notch.hpp"
#include "notch_io.hpp"
#include "notch_pre.hpp"


using namespace std;


TEST_CASE("OneHotEncoder single-column encoding-decoding", "[pre]") {
    Dataset five {{5}, {1}, {4}, {3}, {2}}; // labels in range 1..5
    OneHotEncoder enc {five};
    Dataset fiveEncoded = enc.apply(five);
    // check expected encoding:
    Dataset expected {{0, 0, 0, 0, 1},
                      {1, 0, 0, 0, 0},
                      {0, 0, 0, 1, 0},
                      {0, 0, 1, 0, 0},
                      {0, 1, 0, 0, 0}};
    for (size_t i = 0; i < fiveEncoded.size(); ++i) {
        CHECK(fiveEncoded[i].size() == 5u);
        float sum = 0;
        for(size_t j = 0; j < fiveEncoded[i].size(); ++j) {
            sum += fiveEncoded[i][j];
            CHECK(fiveEncoded[i][j] == expected[i][j]);
        }
        CHECK(sum == 1.0);
    }
    // check decoding
    for (size_t i = 0; i < fiveEncoded.size(); ++i) {
        auto x = enc.unapply(fiveEncoded[i]);
        CHECK(x.size() == five[i].size());
        for (size_t j = 0; j < x.size(); ++j) {
            CHECK(x[j] == five[i][j]);
        }
    }
}

TEST_CASE("OneHotEncoder two-column encoding-decoding", "[pre]") {
    Dataset twocols {{1,10},{3,20},{2,20},{4,30}}; // 4 and 3 distinct values
    OneHotEncoder enc(twocols);
    Dataset encoded = enc.apply(twocols);
    // check expected encoding:
    Dataset expected {{1, 0, 0, 0, 1, 0, 0},
                      {0, 0, 1, 0, 0, 1, 0},
                      {0, 1, 0, 0, 0, 1, 0},
                      {0, 0, 0, 1, 0, 0, 1}};
    size_t col1_code_size = 4;
    size_t col2_code_size = 3;
    for (size_t i = 0; i < encoded.size(); ++i) {
        CHECK(encoded[i].size() == (col1_code_size + col2_code_size));
        float sum = 0;
        for(size_t j = 0; j < encoded[i].size(); ++j) {
            sum += encoded[i][j];
            CHECK(encoded[i][j] == expected[i][j]);
        }
        CHECK(sum == 2.0); // 2 columns
    }
    // check decoding
    for (size_t i = 0; i < encoded.size(); ++i) {
        auto x = enc.unapply(encoded[i]);
        CHECK(x.size() == twocols[i].size());
        for (size_t j = 0; j < x.size(); ++j) {
            CHECK(x[j] == twocols[i][j]);
        }
    }
}

TEST_CASE("OneHotEncoder column selection (negative index)", "[pre]") {
    Dataset twocols {{1,10},{3,20},{2,20},{4,30}}; // 4 and 3 distinct values
    OneHotEncoder enc(twocols, {-2}); // the first (the last but one) column
    Dataset encoded = enc.apply(twocols);
    // check expected encoding:
    Dataset expected {{1, 0, 0, 0, 10},
                      {0, 0, 1, 0, 20},
                      {0, 1, 0, 0, 20},
                      {0, 0, 0, 1, 30}};
    size_t col1_code_size = 4;
    for (size_t i = 0; i < encoded.size(); ++i) {
        CHECK(encoded[i].size() == col1_code_size + 1);
        float sum = 0;
        for(size_t j = 0; j < encoded[i].size(); ++j) {
            CHECK(encoded[i][j] == expected[i][j]);
            if (j < col1_code_size) {
                sum += encoded[i][j];
            }
        }
        CHECK(sum == 1.0); // 1 column
    }
    // check decoding
    for (size_t i = 0; i < encoded.size(); ++i) {
        auto x = enc.unapply(encoded[i]);
        CHECK(x.size() == twocols[i].size());
        for (size_t j = 0; j < x.size(); ++j) {
            CHECK(x[j] == twocols[i][j]);
        }
    }
}

TEST_CASE("SquareAugmented apply and unapply", "[pre]") {
    Dataset x1 {{1, 2}, {3, 4}};
    Dataset expected {{1, 2, 1, 4}, {3, 4, 9, 16}};
    Dataset x2 = SquareAugmented().apply(x1);
    // check apply
    CHECK(x2.size() == expected.size());
    for (size_t i = 0; i < x2.size(); ++i) {
        CHECK(x2[i].size() == expected[i].size());
        for(size_t j = 0; j < x2[i].size(); ++j) {
            CHECK(x2[i][j] == expected[i][j]);
        }
    }
    // check unapply
    for (size_t i = 0; i < x2.size(); ++i) {
        auto x = SquareAugmented().unapply(x2[i]);
        CHECK(x.size() == x1[i].size());
        for (size_t j = 0; j < x.size(); ++j) {
            CHECK(x[j] == x1[i][j]);
        }
    }
}

TEST_CASE("Dataset inputDim() changes after transform", "[pre]") {
    LabeledDataset d {{{1, 2}, {1, 2, 3}}}; // 2 in, 3 out
    CHECK(d.inputDim() == 2u);
    CHECK(d.outputDim() == 3u);
    SquareAugmented square;
    d.apply(square);
    CHECK(d.inputDim() == 2*2u);
    CHECK(d.outputDim() == 3u);
}
