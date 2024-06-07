#pragma once

namespace cart::check {
/**
 * @brief Utility method to check if the cart::util::copyToShared method in cuda.cuh works
 *
 * @param logger
 */
void checkIfCopyWorks(log4cxx::LoggerPtr &logger);
}  // namespace cart::check