#include "logging.hpp"

#include <log4cxx/basicconfigurator.h>
#include <log4cxx/fileappender.h>
#include <log4cxx/helpers/pool.h>
#include <log4cxx/logger.h>
#include <log4cxx/simplelayout.h>

#include "log4cxx/consoleappender.h"

namespace cart {
void configureLogging(const std::string& logFile) {
    log4cxx::FileAppender* fileAppender = new log4cxx::FileAppender(log4cxx::LayoutPtr(new log4cxx::SimpleLayout()), logFile, false);
    log4cxx::ConsoleAppender* consoleAppender = new log4cxx::ConsoleAppender(log4cxx::LayoutPtr(new log4cxx::SimpleLayout()));

    log4cxx::helpers::Pool p;
    fileAppender->activateOptions(p);

    log4cxx::BasicConfigurator::configure(log4cxx::AppenderPtr(fileAppender));
    log4cxx::BasicConfigurator::configure(log4cxx::AppenderPtr(consoleAppender));

#ifdef CARTSLAM_DEBUG
    log4cxx::Logger::getRootLogger()->setLevel(log4cxx::Level::getDebug());
    LOG4CXX_INFO(log4cxx::Logger::getRootLogger(), "Debug mode enabled");
#else
    log4cxx::Logger::getRootLogger()->setLevel(log4cxx::Level::getInfo());
#endif
}

log4cxx::LoggerPtr getLogger(const std::string& name) {
    auto logger = log4cxx::Logger::getLogger(name);

#ifdef CARTSLAM_DEBUG
    logger->setLevel(log4cxx::Level::getDebug());
#else
    logger->setLevel(log4cxx::Level::getInfo());
#endif

    return logger;
}
}  // namespace cart