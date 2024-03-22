#include "cartslam.hpp"

namespace cart {
SystemRunData::~SystemRunData() {
    boost::lock_guard<boost::mutex> lock(this->dataMutex);
    for (auto data : this->data) {
        free(data.second);  // Use free since these are void*
    }

    delete this->dataElement;
}

void SystemRunData::insertData(MODULE_RETURN_VALUE_PAIR data) {
    LOG4CXX_INFO(this->system->logger, "Inserting data with key " << std::quoted(data.first));
    boost::lock_guard<boost::mutex> lock(this->dataMutex);
    this->data.insert(data);
    LOG4CXX_DEBUG(this->system->logger, "Notifying all");
    this->dataCondition.notify_all();
}

bool SystemRunData::isComplete() {
    return this->complete;
}

void SystemRunData::markAsComplete() {
    this->complete = true;
}

SystemRunData* SystemRunData::getRelativeRun(const int8_t offset) {
    return this->system->getRunById(this->id + offset);
}

boost::future<MODULE_RETURN_VALUE> SyncWrapperSystemModule::run(System& system, SystemRunData& data) {
    boost::packaged_task<MODULE_RETURN_VALUE> task([this, &system, &data] {
        auto value = this->runInternal(system, data);
        LOG4CXX_DEBUG(this->logger, "Sync wrapper: Module " << this->name << " has finished");
        if (value) {
            LOG4CXX_DEBUG(this->logger, "Sync wrapper: Module " << this->name << " has data with key " << value->first);
        }

        return value;
    });

    auto future = task.get_future();
    boost::asio::post(system.threadPool, boost::move(task));
    return future;
}

System::System(DataSource* source) : dataSource(source) {
    this->logger = getLogger("System");
}

System::~System() {
    for (auto module : this->modules) {
        delete module;
    }

    delete this->dataSource;
}

void System::addModule(SystemModule* module) {
    this->modules.push_back(module);
    LOG4CXX_INFO(this->logger, "Added module " << module->name);
}

uint8_t System::getActiveRunCount() {
    uint8_t count = 0;

    for (auto run : this->runs) {
        if (!run->isComplete()) {
            count++;
        }
    }

    return count;
}

SystemRunData* System::startNewRun(cv::cuda::Stream& stream) {
    boost::unique_lock<boost::mutex> lock(this->runMutex);
    auto previousRunId = this->runId - 1;
    auto data = new SystemRunData(this->runId++, this, this->dataSource->getNext(stream));
    // Wait for a run to complete if we have reached the limit, and make sure we maintain the correct order
    while (this->getActiveRunCount() >= CARTSLAM_CONCURRENT_RUN_LIMIT || (this->runs.size() > 0 && this->runs[this->runs.size() - 1]->id != previousRunId)) {
        LOG4CXX_DEBUG(this->logger, "Waiting for a run to complete");
        this->runCondition.wait(lock);
    }

    this->runs.push_back(data);

    if (this->runs.size() > CARTSLAM_RUN_RETENTION) {
        LOG4CXX_DEBUG(this->logger, "Deleting old run with ID " << this->runs[0]->id);
        delete this->runs[0];
        this->runs.erase(this->runs.begin());
    }

    return data;
}

SystemRunData* System::getRunById(const uint32_t id) {
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

    SystemRunData* runData = this->startNewRun(stream);
    LOG4CXX_DEBUG(this->logger, "Starting new run with id " << runData->id);

    std::vector<boost::future<void>> moduleFutures;

    for (auto module : this->modules) {
        LOG4CXX_DEBUG(this->logger, "Running module " << std::quoted(module->name));
        boost::future<MODULE_RETURN_VALUE> future;

        if (module->dependsOn.size() > 0) {
            // TODO: Test this
            LOG4CXX_DEBUG(this->logger, "Module " << std::quoted(module->name) << " has dependencies");
            std::vector<boost::future<void*>> dependencies;

            for (auto dependency : module->dependsOn) {
                LOG4CXX_DEBUG(this->logger, "Getting dependency " << std::quoted(dependency));
                dependencies.push_back(runData->getDataAsync<void>(dependency));
            }

            future = boost::when_all(dependencies.begin(), dependencies.end())
                         .then([this, runData, module](auto) {
                             LOG4CXX_DEBUG(this->logger, "All dependencies have been resolved for module " << std::quoted(module->name));
                             return module->run(*this, *runData);
                         })
                         .unwrap();
        } else {
            LOG4CXX_DEBUG(this->logger, "Module " << std::quoted(module->name) << " has no dependencies");
            future = module->run(*this, *runData);
        }

        LOG4CXX_DEBUG(this->logger, "Module " << std::quoted(module->name) << " has been submitted for execution");

        moduleFutures.push_back(future.then([this, runData](boost::future<MODULE_RETURN_VALUE> future) {
            auto data = future.get();
            LOG4CXX_DEBUG(this->logger, "Got result from module");

            if (data) {
                LOG4CXX_DEBUG(this->logger, "Got data with key " << std::quoted(data->first) << ". Inserting into run data.");
                runData->insertData(*data);
            }
        }));
    }

    LOG4CXX_DEBUG(this->logger, "All modules have been submitted for execution");
    log4cxx::LoggerPtr logger = this->logger;
    return boost::when_all(moduleFutures.begin(), moduleFutures.end()).then([this, runData, logger](auto) {
        runData->markAsComplete();
        LOG4CXX_DEBUG(logger, "Run with ID " << runData->id << " has completed");

        boost::lock_guard<boost::mutex> lock(this->runMutex);
        this->runCondition.notify_one();
    });
}

}  // namespace cart