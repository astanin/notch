#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <algorithm>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <string>

#include "notch.hpp"


using namespace std;


/// FullyConnectedLayer_Test breaks encapsulation of FullyConnectedLayer to
/// explore its inner state.
class FullyConnectedLayer_Test : public FullyConnectedLayer {
public:
    FullyConnectedLayer_Test(size_t in, size_t out, const ActivationFunction &af)
        : FullyConnectedLayer(in, out, af) {}

    Weights &getWeights() { return weights; }
    Weights &getBias() { return bias; }
    Array &getInducedLocalField() { return inducedLocalField; }
    Array &getActivationGrad() { return activationGrad; }
    Array &getLocalGrad() { return localGrad; }
    Weights &getWeightsCorrections() { return weightCorrections; }
    Weights &getBiasCorrections() { return biasCorrections; }

    shared_ptr<Array> getLastInputs() { return lastInputs; }
    shared_ptr<Array> getLastOutputs() { return lastOutputs; }
    shared_ptr<BackpropResult> getThisBPR() { return thisBPR; };
    shared_ptr<BackpropResult> getNextBPR() { return nextBPR; };
    bool getBuffersReadyFlag() { return buffersAreReady; };

};

#define CHECK_ARRAY_IS_INITIALIZED(name, arr_expr, expected_size) do { \
    Array &(name) = arr_expr; \
    auto is_zero = [](float x) { return x == Approx(0.0); }; \
    CHECK( (name).size() == (expected_size) ); \
    CHECK( all_of(begin(name), end(name), is_zero) ); \
} while(0)

TEST_CASE( "FullyConnectedLayer construction", "[core]" ) {
    size_t n_in = 3;
    size_t n_out = 2;
    FullyConnectedLayer_Test fc(n_in, n_out, linearActivation);

    // initialization on construction:
    CHECK_ARRAY_IS_INITIALIZED(weights, fc.getWeights(), n_in*n_out);
    CHECK_ARRAY_IS_INITIALIZED(bias, fc.getBias(), n_out);
    CHECK_ARRAY_IS_INITIALIZED(inducedLocalField, fc.getInducedLocalField(), n_out);
    CHECK_ARRAY_IS_INITIALIZED(activationGrad, fc.getActivationGrad(), n_out);
    CHECK_ARRAY_IS_INITIALIZED(localGrad, fc.getLocalGrad(), n_out);
    CHECK_ARRAY_IS_INITIALIZED(weightsCorrections, fc.getWeightsCorrections(), n_in*n_out);
    CHECK_ARRAY_IS_INITIALIZED(biasCorrections, fc.getBiasCorrections(), n_out);
}

TEST_CASE( "FullyConnectedLayer shared buffers initialization", "[core]" ) {
    size_t n_in = 3;
    size_t n_out = 7;
    size_t n_out_next = 4;
    auto rng = newRNG();
    FullyConnectedLayer_Test fc(n_in, n_out, linearActivation);
    FullyConnectedLayer_Test fc2(n_out, n_out_next, linearActivation);
    CHECK_FALSE(fc.getBuffersReadyFlag()); // not until connectTo()

    fc.connectTo(fc2);
    CHECK(fc.getBuffersReadyFlag()); // now ready
    CHECK(fc.getNextBPR() == fc2.getThisBPR()); // buffers are shared
    CHECK(fc.getLastOutputs() == fc2.getLastInputs());

    // check dimensions of the dynamically allocated arrays
    CHECK_ARRAY_IS_INITIALIZED(lastInputs, *fc.getLastInputs(), n_in);
    CHECK_ARRAY_IS_INITIALIZED(lastOutputs, *fc.getLastOutputs(), n_out);

    // check that dimensions of this layer's BackpropResult
    // match layer's dimensions
    shared_ptr<BackpropResult> bpr = fc.getThisBPR();
    CHECK_ARRAY_IS_INITIALIZED(bpr_propagatedErrorSignals,
         bpr->propagatedErrorSignals, n_in);
    CHECK_ARRAY_IS_INITIALIZED(bpr_weightsCorrections,
         bpr->weightCorrections, n_in * n_out);
    CHECK_ARRAY_IS_INITIALIZED(bpr_biasCorrections,
         bpr->biasCorrections, n_out);

    // check that dimensions of the next layer's BackpropResult
    // match layer's dimensions
    shared_ptr<BackpropResult> next_bpr = fc.getNextBPR();
    CHECK_ARRAY_IS_INITIALIZED(next_bpr_propagatedErrorSignals,
         next_bpr->propagatedErrorSignals, n_out);
    CHECK_ARRAY_IS_INITIALIZED(next_bpr_weightsCorrections,
         next_bpr->weightCorrections, n_out * n_out_next);
    CHECK_ARRAY_IS_INITIALIZED(next_bpr_biasCorrections,
         next_bpr->biasCorrections, n_out_next);

    fc.init(rng, normalXavier);
    CHECK(fc.getNextBPR() == fc2.getThisBPR()); // buffers are still shared
    CHECK(fc.getLastOutputs() == fc2.getLastInputs());
}

TEST_CASE( "FullyConnectedLayer from weights matrix (&&)", "[core]" ) {
    /// three in, two out
    FullyConnectedLayer fc({1, 10, 100, 0.1, 0.01, 0.001}, // weights, row-major
                           {2.5, 5.0}, // bias
                           defaultTanh);
    auto out = *fc.output({1,1,1});
    CHECK(out.size() == 2);
    CHECK(out[0] == Approx(tanh(111 + 2.5)));
    CHECK(out[1] == Approx(tanh(0.111 + 5.0)));
}

TEST_CASE( "FullyConnectedLayer from weights matrix (const&)", "[core]" ) {
    /// three in, two out
    const Weights w = {1, 10, 100, 0.1, 0.01, 0.001}; // weights, row-major
    const Weights bias = {2.5, 5.0}; // bias
    FullyConnectedLayer fc(w, bias, linearActivation);
    auto out = *fc.output({1,1,1});
    CHECK(out.size() == 2);
    CHECK(out[0] == Approx(111 + 2.5));
    CHECK(out[1] == Approx(0.111 + 5.0));
}

TEST_CASE( "FullyConnectedLayer init(weights, bias)", "[core]" ) {
    FullyConnectedLayer fc(2, 1, linearActivation);
    auto out_before = *fc.output({1,1});
    CHECK(out_before.size() == 1);
    CHECK(out_before[0] == Approx(0));
    // init using r-value references
    fc.init({10, 100}, {2});
    auto out1 = *fc.output({1,1});
    CHECK(out1.size() == 1);
    CHECK(out1[0] == Approx(112));
    // init using const references
    const Weights ws = {0.1, 0.01};
    const Weights b = {0};
    fc.init(ws, b);
    auto out2 = *fc.output({1, 2});
    CHECK(out2.size() == 1);
    CHECK(out2[0] == Approx(0.1*1 + 0.01*2));
}



#include "test_notch_io.hpp"
