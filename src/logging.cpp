#include "logging.hpp"

#include <log4cxx/basicconfigurator.h>
#include <log4cxx/consoleappender.h>
#include <log4cxx/fileappender.h>
#include <log4cxx/helpers/pool.h>
#include <log4cxx/logger.h>
#include <log4cxx/patternlayout.h>

namespace cart {
void configureLogging(const std::string& logFile) {
    log4cxx::LayoutPtr consoleLayout(new log4cxx::PatternLayout("%d{yyyy-MM-dd HH:mm:ss} %Y%-5p %c{1}%y - %m%n"));
    log4cxx::LayoutPtr fileLayout(new log4cxx::PatternLayout("%d{yyyy-MM-dd HH:mm:ss} %-5p %c{1} - %m%n"));

    log4cxx::FileAppender* fileAppender = new log4cxx::FileAppender(fileLayout, logFile, false);
    log4cxx::ConsoleAppender* consoleAppender = new log4cxx::ConsoleAppender(consoleLayout);

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