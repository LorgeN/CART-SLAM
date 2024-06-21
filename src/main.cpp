#include <iostream>

#include "cartconfig.hpp"
#include "cartslam.hpp"
#include "logging.hpp"
#include "utils/ui.hpp"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Please provide a config file." << std::endl;
        std::cout << "Usage 1: " << argv[0] << " <config file>" << std::endl;
        std::cout << "Usage 2: " << argv[0] << " <data source config> <module config>" << std::endl;
        return 1;
    }

    cart::configureLogging("app.log");

    auto logger = cart::getLogger("main");

    boost::shared_ptr<cart::System> system;
    try {
        if (argc == 2) {
            system = cart::config::readSystemConfig(argv[1]);
        } else {
            auto dataSource = cart::config::readDataSourceConfig(argv[1]);
            system = boost::make_shared<cart::System>(dataSource);
            cart::config::readModuleConfig(argv[2], system);
        }
    } catch (const std::exception& e) {
        LOG4CXX_ERROR(logger, "Error in configuration: " << e.what());
        return 1;
    }

    auto dataSource = system->getDataSource();

    if (dataSource->isFinished()) {
        LOG4CXX_WARN(cart::getLogger("main"), "The provided data source has no data. Exiting.");
        return 1;
    }

    boost::future<void> last;

    while (!dataSource->isFinished()) {
        if (!dataSource->isNextReady()) {
            continue;
        }

        last = system->run().then([logger](boost::future<void> future) {
            try {
                future.get();
            } catch (const std::exception& e) {
                LOG4CXX_ERROR(logger, "Error in processing: " << e.what());
            }
        });
    }

    LOG4CXX_INFO(logger, "Waiting for last run to finish");

    last.get();

    system->getThreadPool().join();

    cart::ImageThread::getInstance().stop();

    LOG4CXX_INFO(logger, "Finished! Good bye :)");

    return 0;
}