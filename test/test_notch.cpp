#include "catch.hpp"

#include <algorithm>
#include <array>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <string>
#include <tuple>

#include "notch.hpp"
#include "notch_io.hpp"

using namespace std;
using namespace notch;

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

    Array &getInducedLocalField() { return inducedLocalField; }
    Array &getActivationGrad() { return activationGrad; }
    Array &getLocalGrad() { return localGrad; }
    shared_ptr<ALearningPolicy> getPolicy() { return policy->clone(); }
    bool getBuffersReadyFlag() { return shared.ready(); };
};

#define CHECK_ARRAY_IS_INITIALIZED(name, arr_expr, expected_size) do { \
    const Array &(name) = arr_expr; \
    auto is_zero = [](float x) { return x == Approx(0.0); }; \
    CHECK( (name).size() == (expected_size) ); \
    CHECK( all_of(begin(name), end(name), is_zero) ); \
} while(0)

TEST_CASE("FC construction from shape", "[core][fc]") {
    size_t n_in = 3;
    size_t n_out = 2;
    FullyConnectedLayer_Test fc(n_in, n_out, linearActivation);

    // initialization on construction:
    CHECK_ARRAY_IS_INITIALIZED(weights, *fc.getWeights(), n_in*n_out);
    CHECK_ARRAY_IS_INITIALIZED(bias, *fc.getBias(), n_out);
    CHECK_ARRAY_IS_INITIALIZED(inducedLocalField, fc.getInducedLocalField(), n_out);
    CHECK_ARRAY_IS_INITIALIZED(activationGrad, fc.getActivationGrad(), n_out);
    CHECK_ARRAY_IS_INITIALIZED(localGrad, fc.getLocalGrad(), n_out);
    CHECK_FALSE(fc.getInputBuffer());
    CHECK_FALSE(fc.getOutputBuffer());
    CHECK_FALSE(fc.getBuffersReadyFlag());
}

TEST_CASE("FC construction from weights matrix (no-copy)", "[core][fc]") {
    /// three in, two out
    FullyConnectedLayer fc({1.0f, 10.0f, 100.0f, 0.1f, 0.01f, 0.001f}, // weights, row-major
                           {2.5f, 5.0f}, // bias
                           defaultTanh);
    auto &out = fc.output({1,1,1});
    CHECK(out.size() == 2u);
    CHECK(out[0] == Approx(tanh(111 + 2.5)));
    CHECK(out[1] == Approx(tanh(0.111 + 5.0)));
}

TEST_CASE("FC construction from weights matrix (copy)", "[core][fc]") {
    /// three in, two out
    const Array w = {1.0f, 10.0f, 100.0f, 0.1f, 0.01f, 0.001f}; // weights, row-major
    const Array bias = {2.5f, 5.0f}; // bias
    FullyConnectedLayer fc(w, bias, linearActivation);
    auto &out = fc.output({1,1,1});
    CHECK(out.size() == 2u);
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
    SECTION("before cloning") {
        // initially:
        auto &weights1 = *fc1.getWeights();
        auto &weights2 = *fc2.getWeights();
        auto &bias1 = *fc1.getBias();
        auto &bias2 = *fc2.getBias();
        auto out1 = fc1.getOutputBuffer();
        auto in1 = fc2.getInputBuffer();
        CHECK(out1 == in1); // buffers are shared
        CHECK(weights1[0] == weights2[0]); // parameters are the same
        CHECK(weights1[1] == weights2[1]); // parameters are the same
        CHECK(weights1[2] == weights2[2]); // parameters are the same
        CHECK(weights1[3] == weights2[3]); // parameters are the same
        CHECK(bias1[0] == bias2[0]); // parameters are the same
        CHECK(bias1[1] == bias2[1]); // parameters are the same
    }
    SECTION("after cloning") {
        auto &weights1 = *fc1.getWeights();
        auto &bias1 = *fc1.getBias();
        // clone is detached:
        shared_ptr<ABackpropLayer> fc1clone = fc1.clone();
        auto out1 = fc1.getOutputBuffer();
        auto out1clone = fc1clone->getOutputBuffer();
        CHECK_FALSE(out1 == out1clone); // not shared
        // clone updates don't affect the original:
        auto rng = Init::newRNG();
        fc1clone->init(rng, Init::uniformXavier);
        auto &cloneRef = (FullyConnectedLayer&) *fc1clone;
        auto &cloneWeights = *cloneRef.getWeights();
        auto &cloneBias = *cloneRef.getBias();
        CHECK_FALSE(cloneWeights[0] == weights1[0]);
        CHECK_FALSE(cloneWeights[1] == weights1[1]);
        CHECK_FALSE(cloneWeights[2] == weights1[2]);
        CHECK_FALSE(cloneWeights[3] == weights1[3]);
        CHECK_FALSE(cloneBias[0] == bias1[0]);
        CHECK_FALSE(cloneBias[1] == bias1[1]);
        // clone buffers are disconnected
        CHECK_FALSE(fc1clone->getOutputBuffer().get() == fc1.getOutputBuffer().get());
    }
}

/// Initialize all weights equal to the number of inputs.
void dummyInit(std::unique_ptr<RNG> &, Array &weights, int n_in, int) {
    std::generate(std::begin(weights), std::end(weights), [&] {
        return n_in;
    });
}

TEST_CASE("ConvolutionLayer2D default ctor", "[core][conv]") {
    ConvolutionLayer2D<3> conv1(5, 4); // 5x4 image in, 3x3 kernel, 3x2 image out
    CHECK(conv1.tag() == "ConvolutionLayer2D");
    CHECK(conv1.inputDim() == 5*4);
    CHECK(conv1.outputDim() == 3*2);
    auto rng = Init::newRNG();
    conv1.init(rng, dummyInit);
    CHECK((*conv1.getWeights())[0] == 9);
    CHECK((*conv1.getBias())[0] == 9);
}

TEST_CASE("ConvolutionLayer2D forward propagation", "[core][conv]") {
    // convolution layer with a box-filter kernel
    ConvolutionLayer2D<3> conv1(4, 3, {1, 1, 1,
                                       1, 1, 1,
                                       1, 1, 1}, {0});
    Array input {9, 0, 0, 0,
                 0, 0, 0, 0,
                 0, 0, 1, 0};
    auto output = conv1.output(input);
    CHECK(conv1.getInputBuffer()->size() == 4*3);
    CHECK(conv1.getOutputBuffer()->size() == 2*1);
    CHECK(output[0] == 10);
    CHECK(output[1] == 1);
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
    const Array w = {0.01f, 0.1f, -0.1f, -0.01f};
    const Array b = {0.25f, -0.25f};
    // compare this:
    auto fcTanh = make_shared<FullyConnectedLayer_Test>(w, b, scaledTanh);
    // vs a net of
    auto fcLinear = make_shared<FullyConnectedLayer_Test>(w, b, linearActivation);
    auto alTanh = make_shared<ActivationLayer>(2, scaledTanh);
    Net net;
    net.append(fcLinear);
    net.append(alTanh);
    net.append(make_shared<EuclideanLoss>(2));
    // forward propagation
    const Array input = {2, 4};
    const Array &fclOut = fcTanh->output(input);
    const Array &netOut = net.output(input);
    CHECK(fclOut.size() == 2u);
    for (size_t i = 0; i < 2; ++i) {
        CHECK(fclOut[i] == Approx(netOut[i]));
    }
    // backpropagation
    const Array error = { 17, 42 };
    const Array &fcl_bpErrors = fcTanh->backprop(error);
    const Array &net_bpErrors = net.backprop(error);
    CHECK(fcl_bpErrors.size() == net_bpErrors.size());
    for (size_t i = 0; i < 2; ++i) {
        CHECK(fcl_bpErrors[i] == Approx(net_bpErrors[i]));
    }
    auto fcl_dEdW = *fcTanh->getWeightSensitivity();
    auto net_dEdW = *fcLinear->getWeightSensitivity();
    CHECK(fcl_dEdW.size() == net_dEdW.size());
    for (size_t i = 0; i < fcl_dEdW.size(); ++i) {
        CHECK(fcl_dEdW[i] == net_dEdW[i]);
    }
    auto fcl_dEdB = *fcTanh->getBiasSensitivity();
    auto net_dEdB = *fcLinear->getBiasSensitivity();
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
    float e = 0.0001f;
    CHECK(loss == Approx(0.0486).epsilon(e));
    Array lossGrad = layer.backprop();
    REQUIRE(lossGrad.size() == 2u);
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
    float b[3] = {1, 2, 42}; // with an extra element at the end
    CHECK_THROWS(internal::gemv(begin(M), end(M), begin(x), end(x), begin(b), end(b)));
    internal::gemv(begin(M), end(M), begin(x), end(x), begin(b), begin(b)+2);
    CHECK(b[0] == Approx(100*1 + 10*2 + 1*3 + 1));
    CHECK(b[1] == Approx(100*4 + 10*5 + 1*6 + 2));
    CHECK(b[2] == 42); // memory is not modified beyond the end
}

TEST_CASE("emul: element-by-element product between vectors", "[core][math]") {
    float x[3] = {1, 2, 3};
    float y[3] = {10, 100, 1000};
    float z[3] = {0.5, 0.5, 0.5};
    float expectedZ[3] = {10, 200, 3000};
    CHECK_THROWS(internal::emul(begin(x), end(x), begin(y), end(y), begin(z), begin(z)+2));
    internal::emul(begin(x), end(x), begin(y), end(y), begin(z), end(z));
    for (size_t i = 0; i < 3; ++i) {
        CHECK(z[i] == expectedZ[i]);
    }
}

TEST_CASE("dot: vector-vector dot product", "[core][math]") {
    float x[3] = {1, 2, 3};
    float y[4] = {100, 10, 1, 42};
    CHECK_THROWS(internal::dot(begin(x), end(x), begin(y), end(y)));
    float p = internal::dot(begin(x), end(x), begin(y), begin(y)+3);
    CHECK(p == 123);
    CHECK(y[3] == 42); // memory is not modified beyond the end
}

TEST_CASE("dot: vector-vector dot product with strides", "[core][math]") {
    float x[4] = {1, 2, 3, 4};
    float y[3] = {1, 10, 100};
    float p = internal::dot(2, x, 2, y+1, 1);
    CHECK(p == 1*10 + 3*100);
}

TEST_CASE("outer: outer vector-vector product", "[core][math]") {
    float alpha = 0.5;
    float x[2] = {10, 100};
    float y[3] = {2, 4, 8};
    float M[7] = {0, 0, 0, 0, 0, 0, 42}; // with an extra element at the end
    float expectedM[6] = {20*alpha, 40*alpha, 80*alpha, 200*alpha, 400*alpha, 800*alpha};
    CHECK_THROWS(internal::outer(alpha, begin(x), end(x), begin(y), end(y), begin(M), end(M)));
    internal::outer(alpha, begin(x), end(x), begin(y), end(y), begin(M), begin(M) + 6);
    for (size_t i =0; i < 6; ++i) {
        CHECK(M[i] == expectedM[i]);
    }
    CHECK(M[6] == 42); // memory is not modified beyond the end
}

TEST_CASE("scale: multiply vector by a scalar", "[core][math]") {
    float alpha = 10.0;
    float x[2] = {2, 4};
    float y[3] = {0, 0, 42}; // with an extra element at the end
    float expectedY[2] = {20, 40};
    CHECK_THROWS(internal::scale(alpha, begin(x), end(x), begin(y), end(y)));
    internal::scale(alpha, begin(x), end(x), begin(y), begin(y)+2);
    for (size_t i =0; i < 2; ++i) {
        CHECK(y[i] == expectedY[i]);
    }
    CHECK(y[2] == 42); // memory is not modified beyond the end
}

TEST_CASE("scale: save output to the same vector", "[core][math]") {
    float alpha = 2;
    float x[3] = {1, 3, 42};
    float expected[2] = {2, 6};
    internal::scale(alpha, begin(x), begin(x)+2, begin(x), begin(x)+2);
    for (size_t i =0; i < 2; ++i) {
        CHECK(x[i] == expected[i]);
    }
    CHECK(x[2] == 42); // memory is not modified beyond the end
}

TEST_CASE("scaleAdd: multiply vector by a scalar and add to another vector", "[core][math]") {
    float alpha = 10.0;
    float x[2] = {2, 4};
    float y[3] = {1, 1, 42}; // with an extra element at the end
    float expectedY[2] = {20+1, 40+1};
    CHECK_THROWS(internal::scaleAdd(alpha, begin(x), end(x), begin(y), end(y)));
    internal::scaleAdd(alpha, begin(x), end(x), begin(y), begin(y)+2);
    for (size_t i =0; i < 2; ++i) {
        CHECK(y[i] == expectedY[i]);
    }
    CHECK(y[2] == 42); // memory is not modified beyond the end
}

#ifdef NOTCH_USE_CBLAS
#ifdef NOTCH_GENERATE_NOBLAS_CODE

TEST_CASE("noblas_gemv: matrix-vector product b = M*x + b", "[core][noblas][math]") {
    float M[6] = {1, 2, 3, 4, 5, 6}; // row-major 3x2
    float x[3] = {100, 10, 1};
    float b[3] = {1, 2, 42}; // with an extra element at the end
    CHECK_THROWS(internal::noblas_gemv(begin(M), end(M), begin(x), end(x), begin(b), end(b)));
    internal::noblas_gemv(begin(M), end(M), begin(x), end(x), begin(b), begin(b)+2);
    CHECK(b[0] == Approx(100*1 + 10*2 + 1*3 + 1));
    CHECK(b[1] == Approx(100*4 + 10*5 + 1*6 + 2));
    CHECK(b[2] == 42); // memory is not modified beyond the end
}

TEST_CASE("noblas_emul: element-by-element product between vectors", "[core][noblas][math]") {
    float x[3] = {1, 2, 3};
    float y[3] = {10, 100, 1000};
    float z[3] = {0.5, 0.5, 0.5};
    float expectedZ[3] = {10, 200, 3000};
    CHECK_THROWS(internal::noblas_emul(begin(x), end(x), begin(y), end(y), begin(z), begin(z)+2));
    internal::noblas_emul(begin(x), end(x), begin(y), end(y), begin(z), end(z));
    for (size_t i = 0; i < 3; ++i) {
        CHECK(z[i] == expectedZ[i]);
    }
}

TEST_CASE("noblas_dot: vector-vector dot product", "[core][noblas][math]") {
    float x[3] = {1, 2, 3};
    float y[4] = {100, 10, 1, 42};
    CHECK_THROWS(internal::noblas_dot(begin(x), end(x), begin(y), end(y)));
    float p = internal::noblas_dot(begin(x), end(x), begin(y), begin(y)+3);
    CHECK(p == 123);
    CHECK(y[3] == 42); // memory is not modified beyond the end
}

TEST_CASE("noblas_dot: vector-vector dot product with strides", "[core][noblas][math]") {
    float x[4] = {1, 2, 3, 4};
    float y[3] = {1, 10, 100};
    float p = internal::noblas_dot(2, x, 2, y+1, 1);
    CHECK(p == 1*10 + 3*100);
}

TEST_CASE("noblas_outer: outer vector-vector product", "[core][noblas][math]") {
    float alpha = 0.5;
    float x[2] = {10, 100};
    float y[3] = {2, 4, 8};
    float M[7] = {0, 0, 0, 0, 0, 0, 42}; // with an extra element at the end
    float expectedM[6] = {20*alpha, 40*alpha, 80*alpha, 200*alpha, 400*alpha, 800*alpha};
    CHECK_THROWS(internal::noblas_outer(alpha, begin(x), end(x), begin(y), end(y), begin(M), end(M)));
    internal::noblas_outer(alpha, begin(x), end(x), begin(y), end(y), begin(M), begin(M) + 6);
    for (size_t i =0; i < 6; ++i) {
        CHECK(M[i] == expectedM[i]);
    }
    CHECK(M[6] == 42); // memory is not modified beyond the end
}

TEST_CASE("noblas_scale: multiply vector by a scalar", "[core][noblas][math]") {
    float alpha = 10.0;
    float x[2] = {2, 4};
    float y[3] = {0, 0, 42}; // with an extra element at the end
    float expectedY[2] = {20, 40};
    CHECK_THROWS(internal::noblas_scale(alpha, begin(x), end(x), begin(y), end(y)));
    internal::noblas_scale(alpha, begin(x), end(x), begin(y), begin(y)+2);
    for (size_t i =0; i < 2; ++i) {
        CHECK(y[i] == expectedY[i]);
    }
    CHECK(y[2] == 42); // memory is not modified beyond the end
}

TEST_CASE("noblas_scale: save output to the same vector", "[core][noblas][math]") {
    float alpha = 2;
    float x[3] = {1, 3, 42};
    float expected[2] = {2, 6};
    internal::noblas_scale(alpha, begin(x), begin(x)+2, begin(x), begin(x)+2);
    for (size_t i =0; i < 2; ++i) {
        CHECK(x[i] == expected[i]);
    }
    CHECK(x[2] == 42); // memory is not modified beyond the end
}

TEST_CASE("noblas_scaleAdd: multiply vector by a scalar and add to another vector", "[core][noblas][math]") {
    float alpha = 10.0;
    float x[2] = {2, 4};
    float y[3] = {1, 1, 42}; // with an extra element at the end
    float expectedY[2] = {20+1, 40+1};
    CHECK_THROWS(internal::noblas_scaleAdd(alpha, begin(x), end(x), begin(y), end(y)));
    internal::noblas_scaleAdd(alpha, begin(x), end(x), begin(y), begin(y)+2);
    for (size_t i =0; i < 2; ++i) {
        CHECK(y[i] == expectedY[i]);
    }
    CHECK(y[2] == 42); // memory is not modified beyond the end
}

#endif
#endif


TEST_CASE("conv2d with a 3x3 kernel", "[core][math][conv]") {
    array<float, 5*4> input;
    iota(begin(input), end(input), 0); // input: 0, 1, ... 5*4-1

    array<float, 3*2 + 1> output;
    output[3*2] = 42;

    SECTION("Sobel X-direction") {
        array<float, 3*3> kernel {1, 0, -1, 2, 0, -2, 1, 0 , -1};
        internal::conv2d<3>(begin(input), 5, 4, begin(kernel), begin(output));
        for (size_t i =0; i < 3*2; ++i) {
            CHECK(output[i] == 2*4); // derivative in X is constant
        }
        CHECK(output[3*2] == 42); // memory is not modified beyond the end
    }

    SECTION("Sobel Y-direction") {
        size_t imageW = 5;
        array<float, 3*3> kernel {1, 2, 1, 0, 0, 0, -1, -2, -1};
        internal::conv2d<3>(begin(input), 5, 4, begin(kernel), begin(output));
        for (size_t i =0; i < 3*2; ++i) {
            CHECK(output[i] == imageW*2*4); // derivative in Y is constant
        }
        CHECK(output[3*2] == 42); // memory is not modified beyond the end
    }

    SECTION("Laplace 2D approximation") {
        array<float, 3*3> kernel {0, 1, 0, 1, -4, 1, 0, 1, 0};
        internal::conv2d<3>(begin(input), 5, 4, begin(kernel), begin(output));
        for (size_t i =0; i < 3*2; ++i) {
            CHECK(output[i] == 0); // it's a plane
        }
        CHECK(output[3*2] == 42); // memory is not modified beyond the end
    }
}

/* This test is based on the backpropagation example by Dan Ventura
 * http://axon.cs.byu.edu/Dan/478/misc/BP.example.pdf */
TEST_CASE("backprop example", "[core][math][fc][mlp]") {
    // initialize network weights as in the example
    Array weights1 {0.23f, -0.79f, 0.1f, 0.21f};
    Array bias1 {0, 0};
    Array weights2 {-0.12f, -0.88f};
    Array bias2 {0};
    auto layer1 = make_shared<FullyConnectedLayer_Test>(weights1, bias1, logisticActivation);
    auto layer2 = make_shared<FullyConnectedLayer_Test>(weights2, bias2, logisticActivation);
    Net mlp;
    mlp.append(layer1);
    mlp.append(layer2);
    mlp.append(make_shared<EuclideanLoss>(1));
    // training example: (0.3, 0.7) -> 0.0
    Array in {0.3f, 0.7f};
    Array expected {0.0f};

    SECTION("precomputed errors") {
        // forward propagation
        auto &actual_out = mlp.output(in);
        float expected_out = 0.37178f;
        CHECK(actual_out[0] == Approx(expected_out).epsilon(0.0002));
        // backpropagation
        auto error = expected - actual_out;
        auto &bpError = mlp.backprop(error);
        // check calculated weight sensitivity at the bottom layer:
        const Array &actual_dEdw = *layer1->getWeightSensitivity();
        const Array expected_dEdw {-7.3745e-4f, -1.7207e-3f, -5.6863e-3f, -1.3268e-2f};
        for (size_t i = 0; i < 4; ++i) {
            CHECK(actual_dEdw[i] == Approx(expected_dEdw[i]).epsilon(0.0002));
        }
    }

    SECTION("errors from ALossLayer") {
        // forward propagation
        mlp.outputWithLoss(in, expected);
        // backpropagation
        mlp.backprop(); // magic! (EuclideanLoss does all the work)
        // check calculated weight sensitivity at the bottom layer:
        const Array &actual_dEdw = *layer1->getWeightSensitivity();
        const Array expected_dEdw {-7.3745e-4f, -1.7207e-3f, -5.6863e-3f, -1.3268e-2f};
        for (size_t i = 0; i < 4; ++i) {
            CHECK(actual_dEdw[i] == Approx(expected_dEdw[i]).epsilon(0.0002));
        }
    }
}

TEST_CASE("FixedRate: delta rule policy", "[core][train]") {
    float eta = 0.5;
    FixedRate policy(eta);
    Array dEdw { 10, 20, 40 }; // weight updates
    Array dEdb { -20, -10 }; // bias updates
    Array weights { 100, 200, 400 };
    Array oldWeights = weights;
    Array bias { 100, 200 };
    Array oldBias = bias;
    policy.update(weights, bias, dEdw, dEdb);
    for (size_t i = 0; i < weights.size(); ++i) {
        CHECK(weights[i] == oldWeights[i] - eta*dEdw[i]);
    }
    for (size_t i = 0; i < bias.size(); ++i) {
        CHECK(bias[i] == oldBias[i] - eta*dEdb[i]);
    }
}

TEST_CASE("FixedRate with momentum: generalized delta rule policy", "[core][train]") {
    float eta = 0.1f;
    float momentum = 0.5f;
    FixedRate policy(eta, momentum);
    Array weights { 100, 200, 400 };
    Array bias { 100, 200 };
    Array oldWeights = weights;
    Array oldBias = bias;
    // weight and bias updates
    Array dEdw { 10, 20, 40 };
    Array dEdb { -20, -10 };
    // first update, no momentum yet
    policy.update(weights, bias, dEdw, dEdb);
    for (size_t i = 0; i < weights.size(); ++i) {
        CHECK(weights[i] == oldWeights[i] - eta*dEdw[i]);
    }
    for (size_t i = 0; i < bias.size(); ++i) {
        CHECK(bias[i] == oldBias[i] - eta*dEdb[i]);
    }
    // second update, momentum is starting to take effect
    oldWeights = weights;
    oldBias = bias;
    policy.update(weights, bias, dEdw, dEdb);
    for (size_t i = 0; i < weights.size(); ++i) {
        CHECK(weights[i] == oldWeights[i] - (momentum+1)*eta*dEdw[i]);
    }
    for (size_t i = 0; i < bias.size(); ++i) {
        CHECK(bias[i] == oldBias[i] - (momentum+1)*eta*dEdb[i]);
    }
}

TEST_CASE("FixedRate with weight-decay", "[core][train]") {
    float eta = 0.1f;
    float momentum = 0.0;
    float decay = 0.1f;
    Array weights { 100, 200 };
    Array bias { 20, 10 };
    Array oldWeights = weights;
    Array oldBias = bias;
    // weight updates
    Array dEdw { 10, 20 };
    Array dEdb { -10, -10 };
    // apply policy without momentum
    FixedRate policy(eta, momentum, decay);
    policy.update(weights, bias, dEdw, dEdb);
    for (size_t i = 0; i < weights.size(); ++i) {
        CHECK(weights[i] == oldWeights[i] - eta*(dEdw[i] + 2*decay*oldWeights[i]));
    }
    for (size_t i = 0; i < bias.size(); ++i) {
        CHECK(bias[i] == oldBias[i] - eta*(dEdb[i] + 2*decay*oldBias[i]));
    }
    // apply policy with momentum
    FixedRate policy2(eta, 0.9f, decay);
    weights = oldWeights;
    bias = oldBias;
    policy2.update(weights, bias, dEdw, dEdb);
    for (size_t i = 0; i < weights.size(); ++i) {
        CHECK(weights[i] == oldWeights[i] - eta*(dEdw[i] + 2*decay*oldWeights[i]));
    }
    for (size_t i = 0; i < bias.size(); ++i) {
        CHECK(bias[i] == oldBias[i] - eta*(dEdb[i] + 2*decay*oldBias[i]));
    }
}

// TODO: write tests for all initialization procedures

