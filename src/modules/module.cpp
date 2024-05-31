#include "modules/module.hpp"

#include "cartslam.hpp"

namespace cart {

boost::future<system_data_t> SyncWrapperSystemModule::run(System& system, SystemRunData& data) {
    boost::packaged_task<system_data_t> task([this, &system, &data] {
        LOG4CXX_DEBUG(this->logger, "Running module sync wrapper " << std::quoted(this->name) << " for ID " << data.id);
        auto value = this->runInternal(system, data);
        LOG4CXX_DEBUG(this->logger, "Sync wrapper of module " << std::quoted(this->name) << " has completed for ID " << data.id);
        return value;
    });

    auto future = task.get_future();
    LOG4CXX_DEBUG(this->logger, "Submitting wrapper task for ID " << data.id);
    boost::asio::post(system.getThreadPool(), boost::move(task));
    return future;
}

const std::vector<module_dependency_t> SystemModule::getRequiredData() const {
    return this->requiresData;
}

const std::vector<std::string> SystemModule::getProvidedData() const {
    return this->providesData;
}
}  // namespace cart