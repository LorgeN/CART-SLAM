#include "cartslam.hpp"

namespace cart {
SystemRunData::~SystemRunData() {
    boost::unique_lock<boost::mutex> lock(this->dataMutex);
    for (auto data : this->data) {
        free(data.second);  // Use free since these are void*
    }

    delete this->dataElement;
}

template <typename T>
T* SystemRunData::getData(std::string key) {
    boost::unique_lock<boost::mutex> lock(this->dataMutex);
    if (!this->data.count(key)) {
        throw std::invalid_argument("Could not find key");
    }

    return static_cast<T*>(this->data[key]);
}

template <typename T>
boost::future<T*> SystemRunData::getDataAsync(std::string key) {
    boost::promise<T*> promise;

    boost::asio::post(this->system->threadPool, [this, &promise, key] {
        boost::unique_lock<boost::mutex> lock(this->dataMutex);
        while (!this->data.count(key)) {
            this->dataCondition.wait(lock);
        }

        promise.set_value(static_cast<T*>(this->data[key]));
    });

    return promise.get_future();
}

void SystemRunData::insertData(MODULE_RETURN_VALUE_PAIR data) {
    LOG4CXX_INFO(this->system->logger, "Inserting data with key " << std::quoted(data.first));
    boost::unique_lock<boost::mutex> lock(this->dataMutex);
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

SystemRunData* System::startNewRun(cv::cuda::Stream& stream) {
    auto data = new SystemRunData(this->runId++, this, this->dataSource->getNext(stream));
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
    return boost::when_all(moduleFutures.begin(), moduleFutures.end()).then([runData, logger](auto) {
        runData->markAsComplete();
        LOG4CXX_DEBUG(logger, "Run with ID " << runData->id << " has completed");
    });
}

}  // namespace cart