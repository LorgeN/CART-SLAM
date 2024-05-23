#pragma once

#include <log4cxx/basicconfigurator.h>
#include <log4cxx/logmanager.h>
#include <log4cxx/propertyconfigurator.h>

namespace cart {
void configureLogging(const std::string& logDir);

log4cxx::LoggerPtr getLogger(const std::string& name);

const std::string getExceptionMessage(const std::exception& e, unsigned int level = 0);
}  // namespace cart
