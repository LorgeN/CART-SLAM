#include "cartslam.hpp"

namespace cart {
log4cxx::LoggerPtr SystemRunData::getLogger() {
    if (boost::shared_ptr<System> sys = this->system.lock()) {
        return sys->getLogger();
    }

    throw std::runtime_error("System has been destroyed");
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

bool SystemRunData::isComplete() {
    return this->complete;
}

void SystemRunData::markAsComplete() {
    this->complete = true;
}

boost::shared_ptr<SystemRunData> SystemRunData::getRelativeRun(const int8_t offset) {
    if (this->id + offset < 0) {
        throw std::invalid_argument("Index out of range");
    }

    if (boost::shared_ptr<System> sys = this->system.lock()) {
        return sys->getRunById(this->id + offset);
    }

    throw std::runtime_error("System has been destroyed");
}

boost::future<module_result_t> SyncWrapperSystemModule::run(System& system, SystemRunData& data) {
    LOG4CXX_DEBUG(this->logger, "Preparing wrapper task");
    boost::packaged_task<module_result_t> task([this, &system, &data] {
        auto value = this->runInternal(system, data);
        LOG4CXX_DEBUG(this->logger, "Sync wrapper: Module " << this->name << " has finished");
        if (value) {
            LOG4CXX_DEBUG(this->logger, "Sync wrapper: Module " << this->name << " has data with key " << value->first);
        }

        return value;
    });

    auto future = task.get_future();
    LOG4CXX_DEBUG(this->logger, "Submitting wrapper task");
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
    boost::unique_lock<boost::mutex> lock(this->runMutex);
    auto previousRunId = this->runId;
    auto data = boost::make_shared<SystemRunData>(++this->runId, this->weak_from_this(), this->dataSource->getNext(stream));
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
    if (id >= this->runId) {
        throw std::invalid_argument("Index out of range");
    }

    uint32_t firstElementId;
    if (this->runId < CARTSLAM_RUN_RETENTION) {
        firstElementId = 0;
    } else {
        firstElementId = this->runId - CARTSLAM_RUN_RETENTION;
    }

    if (id < firstElementId) {
        throw std::invalid_argument("Index out of range (too old)");
    }

    return this->runs[id - firstElementId];
}

boost::future<void> System::run() {
    if (this->modules.size() == 0) {
        throw std::invalid_argument("No modules have been added to the system");
    }

    LOG4CXX_INFO(this->logger, "Running system");
    cv::cuda::Stream stream;

    // TODO: Sort the modules topologically for more efficient execution order

    boost::shared_ptr<SystemRunData> runData = this->startNewRun(stream);
    LOG4CXX_DEBUG(this->logger, "Starting new run with id " << runData->id);

    std::vector<boost::future<void>> moduleFutures;

    for (auto module : this->modules) {
        auto moduleName = std::quoted(module->name);
        LOG4CXX_DEBUG(this->logger, "Running module " << moduleName);
        boost::future<module_result_t> future;

        if (module->requiresData.size() > 0) {
            // TODO: Test this
            LOG4CXX_DEBUG(this->logger, "Module " << moduleName << " has dependencies");

            future = runData->waitForData(module->requiresData)
                         .then([this, runData, module, moduleName](boost::future<void> future) {
                             try {
                                 future.wait();
                             } catch (const std::exception& e) {
                                 LOG4CXX_ERROR(this->logger, "Error waiting for data: " << e.what());
                                 throw;
                             }

                             LOG4CXX_DEBUG(this->logger, "All dependencies have been resolved for module " << moduleName);
                             return module->run(*this, *runData);
                         })
                         .unwrap();
        } else {
            LOG4CXX_DEBUG(this->logger, "Module " << moduleName << " has no dependencies");
            future = module->run(*this, *runData);
        }

        LOG4CXX_DEBUG(this->logger, "Module " << moduleName << " has been submitted for execution");

        moduleFutures.push_back(future.then([this, runData, moduleName](boost::future<module_result_t> future) {
            module_result_t data;

            try {
                data = future.get();
            } catch (const std::exception& e) {
                LOG4CXX_ERROR(this->logger, "Error running module " << moduleName << ": " << e.what());
                throw;
            }

            LOG4CXX_DEBUG(this->logger, "Module " << moduleName << " has completed");

            if (data) {
                LOG4CXX_DEBUG(this->logger, "Got data with key " << std::quoted(data->first) << ". Inserting into run data.");
                runData->insertData(*data);
            }
        }));
    }

    LOG4CXX_DEBUG(this->logger, "All modules have been submitted for execution");
    log4cxx::LoggerPtr logger = this->logger;
    return boost::when_all(moduleFutures.begin(), moduleFutures.end()).then([this, runData, logger](auto future) {
        try {
            future.wait();
        } catch (const std::exception& e) {
            LOG4CXX_ERROR(logger, "Run with ID " << runData->id << " has failed: " << e.what());
            throw;
        }

        runData->markAsComplete();
        LOG4CXX_DEBUG(logger, "Run with ID " << runData->id << " has completed");

        boost::lock_guard<boost::mutex> lock(this->runMutex);
        this->runCondition.notify_all();
    });
}

}  // namespace cart