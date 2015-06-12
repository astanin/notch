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
    // 0 0 0 0 1
    // 1 0 0 0 0
    // 0 0 0 1 0
    // 0 0 1 0 0
    // 0 1 0 0 0
    for (size_t i = 0; i < fiveEncoded.size(); ++i) {
        CHECK(fiveEncoded[i].size() == 5);
        float sum = 0;
        for(size_t j = 0; j < fiveEncoded[i].size(); ++j) {
            sum += fiveEncoded[i][j];
            if (five[i][0]-1 == j) { // labels start at 1
                CHECK(fiveEncoded[i][j] == 1.0);
            } else {
                CHECK(fiveEncoded[i][j] == 0.0);
            }
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
