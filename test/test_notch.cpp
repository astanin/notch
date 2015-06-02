#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <algorithm>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <string>

#include "notch.hpp"
#include "notch_io.hpp"


using namespace std;

// Abbreviations: FC = FullyConnectedLayer
//                AL = ActivationLayer


/// FullyConnectedLayer_Test breaks encapsulation of FullyConnectedLayer to
/// explore its inner state.
class FullyConnectedLayer_Test : public FullyConnectedLayer {
public:
    FullyConnectedLayer_Test(size_t in, size_t out, const ActivationFunction &af)
        : FullyConnectedLayer(in, out, af) {}

    Array &getWeights() { return weights; }
    Array &getBias() { return bias; }
    Array &getInducedLocalField() { return inducedLocalField; }
    Array &getActivationGrad() { return activationGrad; }
    Array &getLocalGrad() { return localGrad; }
    shared_ptr<ALearningPolicy> getPolicy() { return policy->clone(); }

    shared_ptr<Array> getLastInputs() { return lastInputs; }
    shared_ptr<Array> getLastOutputs() { return lastOutputs; }
    shared_ptr<BackpropResult> getBackpropResult() { return backpropResult; };
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
    CHECK_FALSE(fc.getLastInputs());
    CHECK_FALSE(fc.getLastOutputs());
    CHECK_FALSE(fc.getBackpropResult());
    CHECK_FALSE(fc.getBuffersReadyFlag());
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
    CHECK(fc.getLastOutputs() == fc2.getLastInputs()); // buffers are shared

    // check dimensions of the dynamically allocated arrays
    CHECK_ARRAY_IS_INITIALIZED(lastInputs, *fc.getLastInputs(), n_in);
    CHECK_ARRAY_IS_INITIALIZED(lastOutputs, *fc.getLastOutputs(), n_out);

    // check that dimensions of this layer's BackpropResult
    // match layer's dimensions
    shared_ptr<BackpropResult> bpr = fc.getBackpropResult();
    CHECK_ARRAY_IS_INITIALIZED(bpr_propagatedErrors,
         bpr->propagatedErrors, n_in);
    CHECK_ARRAY_IS_INITIALIZED(bpr_weightsSensitivity,
         bpr->weightSensitivity, n_in * n_out);
    CHECK_ARRAY_IS_INITIALIZED(bpr_biasSensitivity,
         bpr->biasSensitivity, n_out);

    fc.init(rng, normalXavier);
    CHECK(fc.getLastOutputs() == fc2.getLastInputs()); // buffers are still shared
}

TEST_CASE( "FullyConnectedLayer from weights matrix (&&)", "[core]" ) {
    /// three in, two out
    FullyConnectedLayer fc({1, 10, 100, 0.1, 0.01, 0.001}, // weights, row-major
                           {2.5, 5.0}, // bias
                           defaultTanh);
    auto out = *fc.output({1,1,1});
    CHECK(out.size() == 2u);
    CHECK(out[0] == Approx(tanh(111 + 2.5)));
    CHECK(out[1] == Approx(tanh(0.111 + 5.0)));
}

TEST_CASE( "FullyConnectedLayer from weights matrix (const&)", "[core]" ) {
    /// three in, two out
    const Array w = {1, 10, 100, 0.1, 0.01, 0.001}; // weights, row-major
    const Array bias = {2.5, 5.0}; // bias
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
    const Array ws = {0.1, 0.01};
    const Array b = {0};
    fc.init(ws, b);
    auto out2 = *fc.output({1, 2});
    CHECK(out2.size() == 1);
    CHECK(out2[0] == Approx(0.1*1 + 0.01*2));
}

TEST_CASE( "FullyConnectedLayer cloning", "[core]") {
    const Array w = {1, -1, -1, 1}; // weights
    const Array bias = {0, 0};  // bias
    FullyConnectedLayer fc1(w, bias, linearActivation);
    FullyConnectedLayer fc2(w, bias, linearActivation);
    fc1.connectTo(fc2);
    // initially:
    CHECK(fc1.getOutputBuffer() == fc2.getInputBuffer()); // buffers are shared
    CHECK(fc1.getWeights()[0] == fc2.getWeights()[0]); // parameters are the same
    CHECK(fc1.getWeights()[1] == fc2.getWeights()[1]); // parameters are the same
    CHECK(fc1.getWeights()[2] == fc2.getWeights()[2]); // parameters are the same
    CHECK(fc1.getWeights()[3] == fc2.getWeights()[3]); // parameters are the same
    CHECK(fc1.getBias()[0] == fc2.getBias()[0]); // parameters are the same
    CHECK(fc1.getBias()[1] == fc2.getBias()[1]); // parameters are the same
    // clone is detached:
    shared_ptr<ABackpropLayer> fc1clone = fc1.clone();
    shared_ptr<Array> out1 = fc1.getOutputBuffer();
    shared_ptr<Array> out1clone = fc1clone->getOutputBuffer();
    CHECK_FALSE(out1 == out1clone); // not shared
    // clone updates don't affect the original:
    auto rng = newRNG();
    fc1clone->init(rng, uniformXavier);
    CHECK_FALSE(fc1clone->getWeights()[0] == fc1.getWeights()[0]);
    CHECK_FALSE(fc1clone->getWeights()[1] == fc1.getWeights()[1]);
    CHECK_FALSE(fc1clone->getWeights()[2] == fc1.getWeights()[2]);
    CHECK_FALSE(fc1clone->getWeights()[3] == fc1.getWeights()[3]);
    CHECK_FALSE(fc1clone->getBias()[0] == fc1.getBias()[0]);
    CHECK_FALSE(fc1clone->getBias()[1] == fc1.getBias()[1]);
    // clone buffers are disconnected
    CHECK_FALSE(fc1clone->getOutputBuffer().get() == fc1.getOutputBuffer().get());
}

TEST_CASE("AL(tanh) ~ FC(I, tanh)", "[core][activation]") {
    const Array identity = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const Array nobias = {0, 0, 0};
    FullyConnectedLayer fcl(identity, nobias, scaledTanh);
    ActivationLayer al(3, scaledTanh);
    // forward propagation
    const Array input = {-1, 1, 100};
    auto fclOut = fcl.output(input);
    auto alOut = al.output(input);
    CHECK(fclOut->size() == alOut->size());
    for (size_t i = 0; i < 3; ++i) {
        CHECK((*fclOut)[i] == Approx((*alOut)[i]));
    }
    // backpropagation
    const Array target = {1, 1, 1};
    auto fclBP = fcl.backprop(target);
    auto alBP = al.backprop(target);
    CHECK(fclBP->propagatedErrors.size() == alBP->propagatedErrors.size());
    for (size_t i = 0; i < 3; ++i) {
        CHECK(fclBP->propagatedErrors[i] == Approx(alBP->propagatedErrors[i]));
    }
}

TEST_CASE("FC(linear) + AL(tanh) ~ FC(tanh)", "[core][activation]") {
    const Array w = {0.01, 0.1, -0.1, -0.01};
    const Array b = {0.25, -0.25};
    // compare this:
    FullyConnectedLayer fcTanh(w, b, scaledTanh);
    // vs a net of
    auto fcLinear = make_shared<FullyConnectedLayer>(w, b, linearActivation);
    auto alTanh = make_shared<ActivationLayer>(2, scaledTanh);
    Net net;
    net.append(fcLinear);
    net.append(alTanh);
    // forward propagation
    const Array input = {2, 4};
    auto fclOut = fcTanh.output(input);
    auto netOut = net.output(input);
    CHECK(fclOut->size() == netOut->size());
    for (size_t i = 0; i < 2; ++i) {
        CHECK((*fclOut)[i] == Approx((*netOut)[i]));
    }
    // backpropagation
    const Array error = { 17, 42 };
    auto fclBP = fcTanh.backprop(error);
    auto netBP = net.backprop(error);
    auto fclBPErrs = fclBP->propagatedErrors;
    auto netBPErrs = netBP->propagatedErrors;
    CHECK(fclBPErrs.size() == netBPErrs.size());
    for (size_t i = 0; i < 2; ++i) {
        CHECK(fclBPErrs[i] == Approx(netBPErrs[i]));
    }
    auto fcl_dEdW = fclBP->weightSensitivity;
    auto net_dEdW = netBP->weightSensitivity;
    CHECK(fcl_dEdW.size() == net_dEdW.size());
    for (size_t i = 0; i < fcl_dEdW.size(); ++i) {
        CHECK(fcl_dEdW[i] == net_dEdW[i]);
    }
    auto fcl_dEdB = fclBP->biasSensitivity;
    auto net_dEdB = netBP->biasSensitivity;
    CHECK(fcl_dEdB.size() == net_dEdB.size());
    for (size_t i = 0; i < fcl_dEdB.size(); ++i) {
        CHECK(fcl_dEdB[i] == net_dEdB[i]);
    }
}

TEST_CASE( "gemv: matrix-vector product b = M*x + b", "[core][math]") {
    float M[6] = {1, 2, 3, 4, 5, 6}; // row-major 3x2
    float x[3] = {100, 10, 1};
    float b[3] = {1, 2, -1}; // with an extra element at the end
    CHECK_THROWS(gemv(begin(M), end(M), begin(x), end(x), begin(b), end(b)));
    gemv(begin(M), end(M), begin(x), end(x), begin(b), begin(b)+2);
    CHECK(b[0] == Approx(100*1 + 10*2 + 1*3 + 1));
    CHECK(b[1] == Approx(100*4 + 10*5 + 1*6 + 2));
    CHECK(b[2] == -1); // unchanged
}

/* This test is based on the backpropagation example by Dan Ventura
 * http://axon.cs.byu.edu/Dan/478/misc/BP.example.pdf */
TEST_CASE( "backprop example", "[core][math]") {
    // initialize network weights as in the example
    MultilayerPerceptron mlp;
    FullyConnectedLayer layer1({0.23, -0.79, 0.1, 0.21}, {0, 0}, logisticActivation);
    FullyConnectedLayer layer2({-0.12, -0.88}, {0}, logisticActivation);
    mlp.append(shared_ptr<FullyConnectedLayer>(&layer1));
    mlp.append(shared_ptr<FullyConnectedLayer>(&layer2));
    // training example: (0.3, 0.7) -> 0.0
    Array in {0.3, 0.7};
    Array expected {0.0};
    // forward propagation
    auto actual_out = mlp.output({0.3, 0.7});
    float expected_out = 0.37178;
    CHECK((*actual_out)[0] == Approx(expected_out).epsilon(0.0002));
    // backpropagation
    Array error = expected - (*actual_out);
    auto backpropResult = mlp.backprop(error);
    // check calculated weight sensitivity at the bottom layer:
    Array &actual_dEdw = backpropResult->weightSensitivity;
    Array expected_dEdw {-7.3745e-4, -1.7207e-3, -5.6863e-3, -1.3268e-2};
    for (size_t i = 0; i < 4; ++i) {
        CHECK(actual_dEdw[i] == Approx(expected_dEdw[i]).epsilon(0.0002));
    }
}

TEST_CASE( "FixedRate (delta rule) policy", "[core][train]") {
    float eta = 0.5;
    FixedRate policy(eta);
    // weight updates
    Array dEdw { 10, 20, 40 };
    Array weights { 100, 200, 400 };
    Array oldWeights = weights;
    policy.correctWeights(dEdw, weights);
    for (size_t i = 0; i < weights.size(); ++i) {
        CHECK(weights[i] == oldWeights[i] - eta*dEdw[i]);
    }
    // bias updates
    Array dBdw { -20, -10 };
    Array bias { 100, 200 };
    Array oldBias = bias;
    policy.correctBias(dBdw, bias);
    for (size_t i = 0; i < bias.size(); ++i) {
        CHECK(bias[i] == oldBias[i] - eta*dBdw[i]);
    }
}

#include "test_notch_io.hpp"
