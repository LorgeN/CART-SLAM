#pragma once

#include <boost/shared_ptr.hpp>
#include <string>

#include "cartslam.hpp"

namespace cart::config {

boost::shared_ptr<cart::System> readSystemConfig(const std::string path);

boost::shared_ptr<cart::DataSource> readDataSourceConfig(const std::string path);

void readModuleConfig(const std::string path, boost::shared_ptr<cart::System> system);

}  // namespace cart::config