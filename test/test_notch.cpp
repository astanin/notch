#include "catch.hpp"

#include <algorithm>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <string>
#include <tuple>

#include "notch.hpp"
#include "notch_io.hpp"

using namespace std;

// Abbreviations:
// FC  = FullyConnectedLayer
// AL  = ActivationLayer

/// FullyConnectedLayer_Test breaks encapsulation of FullyConnectedLayer to
/// explore its inner state.
class FullyConnectedLayer_Test : public FullyConnectedLayer {
public:
    FullyConnectedLayer_Test(size_t in, size_t out,
                             const Activation &af)
        : FullyConnectedLayer(in, out, af) {}
    FullyConnectedLayer_Test(const Array &weights, const Array &bias,
                             const Activation &af)
        : FullyConnectedLayer(weights, bias, af) {}

    Array &getWeights() { return weights; }
    Array &getBias() { return bias; }
    Array &getInducedLocalField() { return inducedLocalField; }
    Array &getActivationGrad() { return activationGrad; }
    Array &getLocalGrad() { return localGrad; }
    shared_ptr<ALearningPolicy> getPolicy() { return policy->clone(); }
    Array &getWeightSensitivity() { return weightSensitivity; }
    Array &getBiasSensitivity() { return biasSensitivity; }
    bool getBuffersReadyFlag() { return shared.ready(); };
};

#define CHECK_ARRAY_IS_INITIALIZED(name, arr_expr, expected_size) do { \
    Array &(name) = arr_expr; \
    auto is_zero = [](float x) { return x == Approx(0.0); }; \
    CHECK( (name).size() == (expected_size) ); \
    CHECK( all_of(begin(name), end(name), is_zero) ); \
} while(0)

TEST_CASE("FC construction from shape", "[core][fc]") {
    size_t n_in = 3;
    size_t n_out = 2;
    FullyConnectedLayer_Test fc(n_in, n_out, linearActivation);

    // initialization on construction:
    CHECK_ARRAY_IS_INITIALIZED(weights, fc.getWeights(), n_in*n_out);
    CHECK_ARRAY_IS_INITIALIZED(bias, fc.getBias(), n_out);
    CHECK_ARRAY_IS_INITIALIZED(inducedLocalField, fc.getInducedLocalField(), n_out);
    CHECK_ARRAY_IS_INITIALIZED(activationGrad, fc.getActivationGrad(), n_out);
    CHECK_ARRAY_IS_INITIALIZED(localGrad, fc.getLocalGrad(), n_out);
    CHECK_FALSE(fc.getInputBuffer());
    CHECK_FALSE(fc.getOutputBuffer());
    CHECK_FALSE(fc.getBuffersReadyFlag());
}

TEST_CASE("FC construction from weights matrix (no-copy)", "[core][fc]") {
    /// three in, two out
    FullyConnectedLayer fc({1, 10, 100, 0.1, 0.01, 0.001}, // weights, row-major
                           {2.5, 5.0}, // bias
                           defaultTanh);
    auto &out = fc.output({1,1,1});
    CHECK(out.size() == 2u);
    CHECK(out[0] == Approx(tanh(111 + 2.5)));
    CHECK(out[1] == Approx(tanh(0.111 + 5.0)));
}

TEST_CASE("FC construction from weights matrix (copy)", "[core][fc]") {
    /// three in, two out
    const Array w = {1, 10, 100, 0.1, 0.01, 0.001}; // weights, row-major
    const Array bias = {2.5, 5.0}; // bias
    FullyConnectedLayer fc(w, bias, linearActivation);
    auto &out = fc.output({1,1,1});
    CHECK(out.size() == 2);
    CHECK(out[0] == Approx(111 + 2.5));
    CHECK(out[1] == Approx(0.111 + 5.0));
}

TEST_CASE("FC-to-FC shared buffers", "[core][fc]") {
    size_t n_in = 3;
    size_t n_out = 7;
    size_t n_out_next = 4;
    auto rng = Init::newRNG();
    FullyConnectedLayer_Test fc(n_in, n_out, linearActivation);
    FullyConnectedLayer_Test fc2(n_out, n_out_next, linearActivation);
    CHECK_FALSE(fc.getBuffersReadyFlag()); // not until connect()

    connect(fc, fc2);
    CHECK(fc.getBuffersReadyFlag()); // now ready
    CHECK(fc.getOutputBuffer() == fc2.getInputBuffer()); // buffers are shared

    // check dimensions of the dynamically allocated arrays
    CHECK_ARRAY_IS_INITIALIZED(inputBuffer, *fc.getInputBuffer(), n_in);
    CHECK_ARRAY_IS_INITIALIZED(outputBuffer, *fc.getOutputBuffer(), n_out);

    fc.init(rng, Init::normalXavier);
    CHECK(fc.getOutputBuffer() == fc2.getInputBuffer()); // buffers are still shared
}

TEST_CASE("FC cloning", "[core][fc]") {
    const Array w = {1, -1, -1, 1}; // weights
    const Array bias = {0, 0};  // bias
    FullyConnectedLayer fc1(w, bias, linearActivation);
    FullyConnectedLayer fc2(w, bias, linearActivation);
    connect(fc1, fc2);
    // initially:
    auto weights1 = GetWeights<FullyConnectedLayer>::ref(fc1);
    auto weights2 = GetWeights<FullyConnectedLayer>::ref(fc2);
    auto bias1 = GetBias<FullyConnectedLayer>::ref(fc1);
    auto bias2 = GetBias<FullyConnectedLayer>::ref(fc2);
    CHECK(fc1.getOutputBuffer() == fc2.getInputBuffer()); // buffers are shared
    CHECK(fc1.getOutputBuffer() == fc2.getInputBuffer()); // buffers are shared
    CHECK(weights1[0] == weights2[0]); // parameters are the same
    CHECK(weights1[1] == weights2[1]); // parameters are the same
    CHECK(weights1[2] == weights2[2]); // parameters are the same
    CHECK(weights1[3] == weights2[3]); // parameters are the same
    CHECK(bias1[0] == bias2[0]); // parameters are the same
    CHECK(bias1[1] == bias2[1]); // parameters are the same
    // clone is detached:
    shared_ptr<ABackpropLayer> fc1clone = fc1.clone();
    shared_ptr<Array> out1 = fc1.getOutputBuffer();
    shared_ptr<Array> out1clone = fc1clone->getOutputBuffer();
    CHECK_FALSE(out1 == out1clone); // not shared
    // clone updates don't affect the original:
    auto rng = Init::newRNG();
    fc1clone->init(rng, Init::uniformXavier);
    auto &cloneRef = (FullyConnectedLayer&) *fc1clone;
    auto cloneWeights = GetWeights<FullyConnectedLayer>::ref(cloneRef);
    auto cloneBias = GetBias<FullyConnectedLayer>::ref(cloneRef);
    CHECK_FALSE(cloneWeights[0] == weights1[0]);
    CHECK_FALSE(cloneWeights[1] == weights1[1]);
    CHECK_FALSE(cloneWeights[2] == weights1[2]);
    CHECK_FALSE(cloneWeights[3] == weights1[3]);
    CHECK_FALSE(cloneBias[0] == bias1[0]);
    CHECK_FALSE(cloneBias[1] == bias1[1]);
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
    auto &fclOut = fcl.output(input);
    auto &alOut = al.output(input);
    CHECK(fclOut.size() == alOut.size());
    for (size_t i = 0; i < 3; ++i) {
        CHECK(fclOut[i] == Approx(alOut[i]));
    }
    // backpropagation
    const Array target = {1, 1, 1};
    auto &fcl_bpErrors = fcl.backprop(target);
    auto &al_bpErrors = al.backprop(target);
    CHECK(fcl_bpErrors.size() == al_bpErrors.size());
    for (size_t i = 0; i < 3; ++i) {
        CHECK(fcl_bpErrors[i] == Approx(al_bpErrors[i]));
    }
}

TEST_CASE("FC(linear) + AL(tanh) ~ FC(tanh)", "[core][activation]") {
    const Array w = {0.01, 0.1, -0.1, -0.01};
    const Array b = {0.25, -0.25};
    // compare this:
    FullyConnectedLayer_Test fcTanh(w, b, scaledTanh);
    FullyConnectedLayer_Test fcLinear(w, b, linearActivation);
    ActivationLayer alTanh(2, scaledTanh);
    // vs a net of
    Net net;
    net.append(std::shared_ptr<FullyConnectedLayer>(&fcLinear));
    net.append(std::shared_ptr<ActivationLayer>(&alTanh));
    net.append(std::make_shared<EuclideanLoss>(2));
    // forward propagation
    const Array input = {2, 4};
    const Array &fclOut = fcTanh.output(input);
    const Array &netOut = net.output(input);
    CHECK(fclOut.size() == 2);
    for (size_t i = 0; i < 2; ++i) {
        CHECK(fclOut[i] == Approx(netOut[i]));
    }
    // backpropagation
    const Array error = { 17, 42 };
    const Array &fcl_bpErrors = fcTanh.backprop(error);
    const Array &net_bpErrors = net.backprop(error);
    CHECK(fcl_bpErrors.size() == net_bpErrors.size());
    for (size_t i = 0; i < 2; ++i) {
        CHECK(fcl_bpErrors[i] == Approx(net_bpErrors[i]));
    }
    auto fcl_dEdW = fcTanh.getWeightSensitivity();
    auto net_dEdW = fcLinear.getWeightSensitivity();
    CHECK(fcl_dEdW.size() == net_dEdW.size());
    for (size_t i = 0; i < fcl_dEdW.size(); ++i) {
        CHECK(fcl_dEdW[i] == net_dEdW[i]);
    }
    auto fcl_dEdB = fcTanh.getBiasSensitivity();
    auto net_dEdB = fcLinear.getBiasSensitivity();
    CHECK(fcl_dEdB.size() == net_dEdB.size());
    for (size_t i = 0; i < fcl_dEdB.size(); ++i) {
        CHECK(fcl_dEdB[i] == net_dEdB[i]);
    }
}

TEST_CASE("AL cloning", "[core][activation]") {
    FullyConnectedLayer fc1(1, 2, linearActivation);
    ActivationLayer a2(2, logisticActivation);
    FullyConnectedLayer fc3(2, 1, defaultTanh);
    connect(fc1, a2);
    connect(a2, fc3);
    // initially buffers are shared:
    CHECK(fc1.getOutputBuffer() == a2.getInputBuffer());
    CHECK(a2.getOutputBuffer() == fc3.getInputBuffer());
    // but clone is detached:
    shared_ptr<ABackpropLayer> a2clone = a2.clone();
    shared_ptr<Array> in2 = a2.getInputBuffer();
    shared_ptr<Array> in2clone = a2clone->getInputBuffer();
    shared_ptr<Array> out2 = a2.getOutputBuffer();
    shared_ptr<Array> out2clone = a2clone->getOutputBuffer();
    CHECK_FALSE(in2 == in2clone); // not shared
    CHECK_FALSE(out2 == out2clone); // not shared
}

TEST_CASE("EuclideanLoss output", "[core][loss][math]") {
    EuclideanLoss layer(2);
    Array target {1, 1};
    Array error {3, 4};
    Array y = target + error;
    float loss = layer.output(y, target);
    CHECK(loss == Approx(sqrt(error[0]*error[0]+error[1]*error[1])));
    Array lossGrad = layer.backprop();
    REQUIRE(lossGrad.size() == error.size());
    CHECK(lossGrad[0] == -error[0]);
    CHECK(lossGrad[1] == -error[1]);
}

TEST_CASE("FC-to-L2 shared buffers", "[core][fc][loss]") {
    size_t n_in = 3;
    size_t n_out = 7;
    FullyConnectedLayer fc(n_in, n_out, linearActivation);
    EuclideanLoss loss(n_out);
    connect(fc, loss); // buffers are shared
    auto lossBuffer = GetShared<EuclideanLoss>::ref(loss).inputBuffer;
    CHECK(fc.getOutputBuffer() == lossBuffer);
    auto lossClonePtr = loss.clone(); // buffers are not shared
    EuclideanLoss &lossClone = static_cast<EuclideanLoss&>(*lossClonePtr);
    auto lossCloneBuffer = GetShared<EuclideanLoss>::ref(lossClone).inputBuffer;
    CHECK_FALSE(fc.getOutputBuffer() == lossCloneBuffer);
}

TEST_CASE("SoftmaxWithLoss output", "[core][loss][math]") {
    SoftmaxWithLoss layer(2);
    Array target {0, 1};
    Array y {1, 4};
    float loss = layer.output(y, target);
    float e = 0.0001;
    CHECK(loss == Approx(0.0486).epsilon(e));
    Array lossGrad = layer.backprop();
    REQUIRE(lossGrad.size() == 2);
    CHECK(lossGrad[0] == Approx(-0.0474).epsilon(e));
    CHECK(lossGrad[1] == Approx( 0.0474).epsilon(e));
}

TEST_CASE("FC-to-SoftmaxWithLoss shared buffers", "[core][fc][loss]") {
    size_t n_in = 3;
    size_t n_out = 7;
    FullyConnectedLayer fc(n_in, n_out, linearActivation);
    SoftmaxWithLoss loss(n_out);
    connect(fc, loss); // buffers are shared
    auto lossBuffer = GetShared<SoftmaxWithLoss>::ref(loss).inputBuffer;
    CHECK(fc.getOutputBuffer() == lossBuffer);
    auto lossClonePtr = loss.clone(); // buffers are not shared
    SoftmaxWithLoss &lossClone = static_cast<SoftmaxWithLoss&>(*lossClonePtr);
    auto lossCloneBuffer = GetShared<SoftmaxWithLoss>::ref(lossClone).inputBuffer;
    CHECK_FALSE(fc.getOutputBuffer() == lossCloneBuffer);
}

TEST_CASE("HingeLoss output", "[core][loss][math]") {
    HingeLoss hinge;
    {
        float loss = hinge.output({-0.25}, {1.0});
        Array lossGrad = hinge.backprop();
        CHECK(loss == 1.25);
        CHECK(lossGrad[0] == -1.0);
    }
    {
        float loss = hinge.output({1.25}, {1.0});
        Array lossGrad = hinge.backprop();
        CHECK(loss == 0.0);
        CHECK(lossGrad[0] == 0.0);
    }
    {
        float loss = hinge.output({0.5}, {-1.0});
        Array lossGrad = hinge.backprop();
        CHECK(loss == 1.5);
        CHECK(lossGrad[0] == 1.0);
    }
}

TEST_CASE("gemv: matrix-vector product b = M*x + b", "[core][math]") {
    float M[6] = {1, 2, 3, 4, 5, 6}; // row-major 3x2
    float x[3] = {100, 10, 1};
    float b[3] = {1, 2, -1}; // with an extra element at the end
    CHECK_THROWS(gemv(begin(M), end(M), begin(x), end(x), begin(b), end(b)));
    gemv(begin(M), end(M), begin(x), end(x), begin(b), begin(b)+2);
    CHECK(b[0] == Approx(100*1 + 10*2 + 1*3 + 1));
    CHECK(b[1] == Approx(100*4 + 10*5 + 1*6 + 2));
    CHECK(b[2] == -1); // unchanged
}

TEST_CASE("sdot: vector-vector dot product", "[core][math]") {
    float x[3] = {1, 2, 3};
    float y[3] = {100, 10, 1};
    float longY[4] = {100, 10, 1, 0.1};
    CHECK_THROWS(sdot(begin(x), end(x), begin(longY), end(longY)));
    float p = sdot(begin(x), end(x), begin(y), end(y));
    CHECK(p == 123);
}

/* This test is based on the backpropagation example by Dan Ventura
 * http://axon.cs.byu.edu/Dan/478/misc/BP.example.pdf */
TEST_CASE("backprop example with precomputed errors", "[core][math][fc][mlp]") {
    // initialize network weights as in the example
    FullyConnectedLayer_Test layer1({0.23, -0.79, 0.1, 0.21}, {0, 0}, logisticActivation);
    FullyConnectedLayer_Test layer2({-0.12, -0.88}, {0}, logisticActivation);
    Net mlp;
    mlp.append(shared_ptr<FullyConnectedLayer>(&layer1));
    mlp.append(shared_ptr<FullyConnectedLayer>(&layer2));
    mlp.append(std::make_shared<EuclideanLoss>(1));
    // training example: (0.3, 0.7) -> 0.0
    Array in {0.3, 0.7};
    Array expected {0.0};
    // forward propagation
    auto &actual_out = mlp.output({0.3, 0.7});
    float expected_out = 0.37178;
    CHECK(actual_out[0] == Approx(expected_out).epsilon(0.0002));
    // backpropagation
    auto error = expected - actual_out;
    auto &bpError = mlp.backprop(error);
    // check calculated weight sensitivity at the bottom layer:
    Array &actual_dEdw = layer1.getWeightSensitivity();
    Array expected_dEdw {-7.3745e-4, -1.7207e-3, -5.6863e-3, -1.3268e-2};
    for (size_t i = 0; i < 4; ++i) {
        CHECK(actual_dEdw[i] == Approx(expected_dEdw[i]).epsilon(0.0002));
    }
}

TEST_CASE("backprop example with LossLayer", "[core][math][fc][loss][mlp]") {
    // initialize network weights as in the example
    FullyConnectedLayer_Test layer1({0.23, -0.79, 0.1, 0.21}, {0, 0}, logisticActivation);
    FullyConnectedLayer_Test layer2({-0.12, -0.88}, {0}, logisticActivation);
    Net mlp;
    mlp.append(shared_ptr<FullyConnectedLayer>(&layer1));
    mlp.append(shared_ptr<FullyConnectedLayer>(&layer2));
    mlp.append(std::make_shared<EuclideanLoss>(1));
    // training example: (0.3, 0.7) -> 0.0
    Array in {0.3, 0.7};
    Array expected {0.0};
    // forward propagation
    mlp.outputWithLoss(in, expected);
    // backpropagation
    mlp.backprop(); // magic! (EuclideanLoss does all the work)
    // check calculated weight sensitivity at the bottom layer:
    Array &actual_dEdw = layer1.getWeightSensitivity();
    Array expected_dEdw {-7.3745e-4, -1.7207e-3, -5.6863e-3, -1.3268e-2};
    for (size_t i = 0; i < 4; ++i) {
        CHECK(actual_dEdw[i] == Approx(expected_dEdw[i]).epsilon(0.0002));
    }
}

TEST_CASE("FixedRate: delta rule policy", "[core][train]") {
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

TEST_CASE("FixedRate with momentum: generalized delta rule policy", "[core][train]") {
    float eta = 0.1;
    float momentum = 0.9;
    FixedRate policy(eta, momentum);
    Array weights { 100, 200, 400 };
    Array bias { 100, 200 };
    Array oldWeights = weights;
    Array oldBias = bias;
    // weight updates
    Array dEdw { 10, 20, 40 };
    Array dBdw { -20, -10 };
    // first update, no momentum yet
    policy.correctWeights(dEdw, weights);
    policy.correctBias(dBdw, bias);
    for (size_t i = 0; i < weights.size(); ++i) {
        CHECK(weights[i] == oldWeights[i] - eta*dEdw[i]);
    }
    for (size_t i = 0; i < bias.size(); ++i) {
        CHECK(bias[i] == oldBias[i] - eta*dBdw[i]);
    }
    // second update, momentum is starting to take effect
    oldWeights = weights;
    oldBias = bias;
    policy.correctWeights(dEdw, weights);
    policy.correctBias(dBdw, bias);
    for (size_t i = 0; i < weights.size(); ++i) {
        CHECK(weights[i] == oldWeights[i] - (momentum+1)*eta*dEdw[i]);
    }
    for (size_t i = 0; i < bias.size(); ++i) {
        CHECK(bias[i] == oldBias[i] - (momentum+1)*eta*dBdw[i]);
    }
}

TEST_CASE("FixedRate with weight-decay", "[core][train]") {
    float eta = 0.1;
    float momentum = 0.0;
    float decay = 0.1;
    Array weights { 100, 200 };
    Array bias { 20, 10 };
    Array oldWeights = weights;
    Array oldBias = bias;
    // weight updates
    Array dEdw { 10, 20 };
    Array dBdw { -10, -10 };
    // apply policy without momentum
    FixedRate policy(eta, momentum, decay);
    policy.correctWeights(dEdw, weights);
    policy.correctBias(dBdw, bias);
    for (size_t i = 0; i < weights.size(); ++i) {
        CHECK(weights[i] == oldWeights[i] - eta*(dEdw[i] + 2*decay*oldWeights[i]));
    }
    for (size_t i = 0; i < bias.size(); ++i) {
        CHECK(bias[i] == oldBias[i] - eta*(dBdw[i] + 2*decay*oldBias[i]));
    }
    // apply policy with momentum
    FixedRate policy2(eta, 0.9, decay);
    weights = oldWeights;
    bias = oldBias;
    policy2.correctWeights(dEdw, weights);
    policy2.correctBias(dBdw, bias);
    for (size_t i = 0; i < weights.size(); ++i) {
        CHECK(weights[i] == oldWeights[i] - eta*(dEdw[i] + 2*decay*oldWeights[i]));
    }
    for (size_t i = 0; i < bias.size(); ++i) {
        CHECK(bias[i] == oldBias[i] - eta*(dBdw[i] + 2*decay*oldBias[i]));
    }
}

// TODO: write tests for all initialization procedures

