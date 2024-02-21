#include "cartslam.hpp"

namespace cart {
SystemRunData::~SystemRunData() {
    boost::unique_lock<boost::mutex> lock(this->dataMutex);
    for (auto data : this->data) {
        free(data.second);  // Use free since these are void*
    }
}

DataElement* SystemRunData::getDataElement() {
    return this->dataElement;
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

    boost::asio::post(this->system.threadPool, [this, &promise, key] {
        boost::unique_lock<boost::mutex> lock(this->dataMutex);
        while (!this->data.count(key)) {
            this->dataCondition.wait(lock);
        }

        promise.set_value(static_cast<T*>(this->data[key]));
    });

    return promise.get_future();
}

void SystemRunData::insertData(MODULE_RETURN_VALUE_PAIR& data) {
    boost::unique_lock<boost::mutex> lock(this->dataMutex);
    this->data.insert(data);
    this->dataCondition.notify_all();
}

boost::future<MODULE_RETURN_VALUE> SyncWrapperSystemModule::run(System& system, SystemRunData& data) {
    boost::promise<MODULE_RETURN_VALUE> promise;

    boost::asio::post(system.threadPool, [this, &system, &data, &promise] {
        promise.set_value(this->runInternal(system, data));
    });

    return promise.get_future();
}

System::System(DataSource* source) {
    this->dataSource = source;
}

System::~System() {
    for (auto module : this->modules) {
        delete module;
    }

    delete this->dataSource;
}

void System::addModule(SystemModule* module) {
    this->modules.push_back(module);
}

boost::future<void> System::run() {
    std::cout << "Running system" << std::endl;
    cv::cuda::Stream stream;
    auto element = this->dataSource->getNext(stream);

    // TODO: Sort the modules topologically for more efficient execution order

    SystemRunData runData(*this, element);

    std::vector<boost::future<void>> moduleFutures;

    for (auto module : this->modules) {
        boost::future<MODULE_RETURN_VALUE> future;

        if (module->dependsOn.size() > 0) {
            std::vector<boost::future<void*>> dependencies;

            for (auto dependency : module->dependsOn) {
                dependencies.push_back(runData.getDataAsync<void>(dependency));
            }

            future = boost::when_all(dependencies.begin(), dependencies.end())
                         .then([this, &runData, module](auto) {
                             return module->run(*this, runData);
                         })
                         .unwrap();
        } else {
            future = module->run(*this, runData);
        }

        future.then([this, &runData](boost::future<MODULE_RETURN_VALUE> future) {
            auto data = future.get();
            if (data) {
                std::cout << "Got data from " << data->first << std::endl;
                runData.insertData(*data);
            }
        });
    }

    return boost::when_all(moduleFutures.begin(), moduleFutures.end()).then([element](auto) {});
}

}  // namespace cart