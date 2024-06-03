#include "utils/random.hpp"

namespace cart::util {
std::vector<size_t> RandomSampler::operator()(size_t amount) {
    std::vector<size_t> samples;
    samples.reserve(amount);

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, this->maxValue);

    size_t valid_sample = 0;
    while (valid_sample < amount) {
        const size_t idx = dist(rng);
        // Well, this is slow. But typically the sample_size is small.
        if (std::find(samples.begin(), samples.end(), idx) == samples.end()) {
            samples.push_back(idx);
            valid_sample++;
        }
    }

    return samples;
}
}  // namespace cart::util