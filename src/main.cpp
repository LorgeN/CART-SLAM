#include <iostream>

#include "cartslam.hpp"
#include "cartconfig.hpp"
#include "logging.hpp"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Please provide an image file to process." << std::endl;
        return 1;
    }

    cart::configureLogging("app.log");

    auto logger = cart::getLogger("main");

    auto system = cart::config::readSystemConfig(argv[1]);
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

    last.get();

    system->getThreadPool().join();
    return 0;
}