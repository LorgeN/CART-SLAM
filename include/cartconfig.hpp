#pragma once

#include <boost/shared_ptr.hpp>
#include <string>

#include "cartslam.hpp"

namespace cart::config {

boost::shared_ptr<cart::System> readSystemConfig(const std::string path);

}  // namespace cart::config