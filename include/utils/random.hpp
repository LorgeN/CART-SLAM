#pragma once

#include <algorithm>
#include <random>
#include <vector>

namespace cart::util {

class RandomSampler {
   public:
    RandomSampler(const size_t maxValue) : maxValue(maxValue) {}

    std::vector<size_t> operator()(size_t amount);

   private:
    size_t maxValue;
};

}  // namespace cart::util