#include "cartslam.hpp"

namespace cart {
log4cxx::LoggerPtr SystemRunData::getLogger() {
    return this->logger;
}

boost::asio::thread_pool& SystemRunData::getThreadPool() {
    if (boost::shared_ptr<System> sys = this->system.lock()) {
        return sys->getThreadPool();
    }

    throw std::runtime_error("System has been destroyed");
}

log4cxx::LoggerPtr System::getLogger() {
    return this->logger;
}

boost::asio::thread_pool& System::getThreadPool() {
    return this->threadPool;
}

void System::insertGlobalData(const std::string key, boost::shared_ptr<void> data) {
    this->insertData(std::make_pair(key, data));
}

void System::insertGlobalData(system_data_pair_t data) {
    this->insertData(data);
}

bool SystemRunData::isComplete() {
    return this->complete;
}

void SystemRunData::markAsComplete() {
    this->complete = true;
}

boost::shared_ptr<SystemRunData> SystemRunData::getRelativeRun(const int8_t offset) {
    if (this->id + offset <= 0) {
        throw std::invalid_argument("Offset " + std::to_string(offset) + " out of range");
    }

    if (boost::shared_ptr<System> sys = this->system.lock()) {
        return sys->getRunById(this->id + offset);
    }

    throw std::runtime_error("System has been destroyed");
}

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

System::System(boost::shared_ptr<DataSource> source) : dataSource(source) {
    this->logger = cart::getLogger("System");
}

void System::addModule(boost::shared_ptr<SystemModule> module) {
    this->modules.push_back(module);
    LOG4CXX_INFO(this->logger, "Added module " << module->name);
}

uint8_t System::getActiveRunCount() {
    for (uint8_t i = 0; i < this->runs.size(); i++) {
        if (!this->runs[i]->isComplete()) {
            return this->runs.size() - i;
        }
    }

    return 0;
}

boost::shared_ptr<SystemRunData> System::startNewRun(cv::cuda::Stream& stream) {
    boost::unique_lock<boost::shared_mutex> lock(this->runMutex);
    auto previousRunId = this->runId;

    boost::shared_ptr<DataElement> element;

    try {
        element = this->dataSource->getNext(this->logger, stream);
    } catch (const std::exception& e) {
        LOG4CXX_ERROR(this->logger, "Error getting next element: " << e.what());
        throw;
    }

    auto data = boost::make_shared<SystemRunData>(++this->runId, this->weak_from_this(), element);
    // Wait for a run to complete if we have reached the limit, and make sure we maintain the correct order
    while (this->getActiveRunCount() >= CARTSLAM_CONCURRENT_RUN_LIMIT || (this->runs.size() > 0 && this->runs[this->runs.size() - 1]->id != previousRunId)) {
        LOG4CXX_DEBUG(this->logger, "Waiting for a run to complete");
        this->runCondition.wait(lock);
    }

    this->runs.push_back(data);

    if (this->runs.size() > CARTSLAM_RUN_RETENTION) {
        LOG4CXX_DEBUG(this->logger, "Deleting old run with ID " << this->runs[0]->id);
        this->runs.erase(this->runs.begin());
    }

    return data;
}

boost::shared_ptr<SystemRunData> System::getRunById(const uint32_t id) {
    boost::shared_lock_guard<boost::shared_mutex> lock(this->runMutex);
    if (id > this->runId) {
        throw std::invalid_argument("Index " + std::to_string(id) + " out of range (too new)");
    }

    uint32_t firstElementId = this->runs[0]->id;
    if (id < firstElementId) {
        throw std::invalid_argument("Index " + std::to_string(id) + " out of range (too old)");
    }

    return this->runs[id - firstElementId];
}

boost::future<void> System::run() {
    if (this->modules.size() == 0) {
        throw std::invalid_argument("No modules have been added to the system");
    }

    cv::cuda::Stream stream;

    // TODO: Sort the modules topologically for more efficient execution order

    boost::shared_ptr<SystemRunData> runData = this->startNewRun(stream);
    LOG4CXX_INFO(this->logger, "Starting new run with id " << runData->id);

    std::vector<boost::future<void>> moduleFutures;

    for (auto module : this->modules) {
        std::string moduleName = "\"" + module->name + "\"";
        LOG4CXX_INFO(this->logger, "Running module " << moduleName);
        boost::future<system_data_t> future;

        if (module->requiresData.size() > 0) {
            // TODO: Test this
            LOG4CXX_DEBUG(this->logger, "Module " << moduleName << " has dependencies");

            future = runData->waitForData(module->requiresData)
                         .then([this, runData, module, moduleName](boost::future<void> future) {
                             try {
                                 future.get();
                             } catch (const std::exception& e) {
                                 LOG4CXX_ERROR(runData->getLogger(), "Error waiting for data: " << cart::getExceptionMessage(e));
                                 std::throw_with_nested(std::runtime_error("Error waiting for data"));
                             }

                             LOG4CXX_DEBUG(runData->getLogger(), "All dependencies have been resolved for module " << moduleName);
                             return module->run(*this, *runData);
                         })
                         .unwrap();
        } else {
            LOG4CXX_DEBUG(runData->getLogger(), "Module " << moduleName << " has no dependencies");
            future = module->run(*this, *runData);
        }

        LOG4CXX_DEBUG(runData->getLogger(), "Module " << moduleName << " has been submitted for execution");

        moduleFutures.push_back(future.then([this, runData, moduleName](boost::future<system_data_t> future) {
            system_data_t data;

            try {
                data = future.get();
            } catch (const std::exception& e) {
                LOG4CXX_ERROR(runData->getLogger(), "Error running module " << moduleName << ": " << cart::getExceptionMessage(e));
                std::throw_with_nested(std::runtime_error("Error running module " + moduleName + " for run ID " + std::to_string(runData->id)));
            }

            LOG4CXX_INFO(runData->getLogger(), "Module " << moduleName << " has completed for run ID " << runData->id);

            for (auto data : data) {
                LOG4CXX_DEBUG(runData->getLogger(), "Got data with key " << std::quoted(data.first) << " from module " << moduleName << " in run " << runData->id);
                runData->insertData(data);
            }
        }));
    }

    LOG4CXX_DEBUG(runData->getLogger(), "All modules have been submitted for execution for run ID " << runData->id);
    log4cxx::LoggerPtr logger = runData->getLogger();

    return boost::when_all(moduleFutures.begin(), moduleFutures.end()).then([this, runData, logger](auto future) {
        LOG4CXX_DEBUG(logger, "All modules have finished for run with ID " << runData->id);

        try {
            boost::csbl::vector<boost::future<void>> allComplete = future.get();

            for (boost::future<void>& moduleFuture : allComplete) {
                moduleFuture.get();
            }
        } catch (const std::exception& e) {
            LOG4CXX_ERROR(logger, "Run with ID " << runData->id << " has failed: " << cart::getExceptionMessage(e));
            throw;
        }

        runData->markAsComplete();
        LOG4CXX_INFO(logger, "Run with ID " << runData->id << " has completed.");

        boost::lock_guard<boost::shared_mutex> lock(this->runMutex);
        this->runCondition.notify_all();
    });
}

}  // namespace cart