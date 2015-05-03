#ifndef RANDOMGEN_H
#define RANDOMGEN_H

#include <random>
#include <array>      // array
#include <algorithm>  // generate
#include <functional> // ref
#include <memory>     // unique_ptr


using rng_type = std::mt19937;


std::unique_ptr<rng_type> seed_rng() {
    std::random_device rd("/dev/urandom");
    std::array<uint32_t, std::mt19937::state_size> seed_data;
    generate(seed_data.begin(), seed_data.end(), ref(rd));
    std::seed_seq sseq(std::begin(seed_data), std::end(seed_data));
    std::unique_ptr<rng_type> rng(new rng_type());
    rng->seed(sseq);
    return rng;
}


#endif /* ifndef RANDOMGEN_H */
