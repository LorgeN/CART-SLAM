#ifndef CARTSLAM_HPP
#define CARTSLAM_HPP

#define BOOST_THREAD_VERSION 3
#define BOOST_THREAD_PROVIDES_FUTURE
#define BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION
#define BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY
#define BOOST_THREAD_PROVIDES_FUTURE_UNWRAP

#define CARTSLAM_WORKER_THREADS 16
#define CARTSLAM_RUN_RETENTION 10
#define CARTSLAM_CONCURRENT_RUN_LIMIT 6

#define MODULE_NO_RETURN_VALUE std::nullopt
#define MODULE_RETURN_VALUE_PAIR std::pair<std::string, void*>
#define MODULE_RETURN_VALUE std::optional<std::pair<std::string, void*>>

#include <log4cxx/logger.h>

#include <boost/asio/post.hpp>
#include <boost/asio/thread_pool.hpp>
#include <boost/thread.hpp>
#include <boost/thread/future.hpp>
#include <map>
#include <string>

#include "datasource.hpp"
#include "logging.hpp"
#include "opencv2/core/cuda.hpp"

namespace cart {
class System;  // Allow references

class SystemRunData {
   public:
    SystemRunData(uint32_t id, System* system, DataElement* dataElement) : id(id), system(system), dataElement(dataElement) {}
    ~SystemRunData();

    template <typename T>
    T* getData(const std::string key);

    template <typename T>
    boost::future<T*> getDataAsync(const std::string key);

    void insertData(MODULE_RETURN_VALUE_PAIR data);

    void markAsComplete();
    bool isComplete();

    SystemRunData* getRelativeRun(const int8_t offset);

    DataElement* dataElement;
    const uint32_t id;

   private:
    bool complete = false;
    System* system;
    std::map<std::string, void*> data;
    boost::mutex dataMutex;
    boost::condition_variable dataCondition;
};

class SystemModule {
   public:
    SystemModule(const std::string& name) : SystemModule(name, {}){};

    SystemModule(const std::string& name, const std::vector<std::string> dependsOn) : name(name), dependsOn(dependsOn) {
        this->logger = getLogger(name);
    }

    virtual ~SystemModule() = default;
    virtual boost::future<MODULE_RETURN_VALUE> run(System& system, SystemRunData& data) = 0;

    const std::vector<std::string> dependsOn;
    const std::string name;

   protected:
    log4cxx::LoggerPtr logger;
};

class SyncWrapperSystemModule : public SystemModule {
   public:
    SyncWrapperSystemModule(const std::string& name) : SystemModule(name){};

    SyncWrapperSystemModule(const std::string& name, const std::vector<std::string> dependsOn) : SystemModule(name, dependsOn){};

    boost::future<MODULE_RETURN_VALUE> run(System& system, SystemRunData& data) override;
    virtual MODULE_RETURN_VALUE runInternal(System& system, SystemRunData& data) = 0;
};

class System {
   public:
    System(DataSource* dataSource);
    ~System();
    boost::future<void> run();
    void addModule(SystemModule* module);

    SystemRunData* startNewRun(cv::cuda::Stream& stream);
    SystemRunData* getRunById(const uint32_t index);
    uint8_t getActiveRunCount();

    log4cxx::LoggerPtr logger;
    boost::asio::thread_pool threadPool = boost::asio::thread_pool(CARTSLAM_WORKER_THREADS);

   private:
    uint32_t runId = 0;
    DataSource* dataSource;
    std::vector<SystemModule*> modules;
    std::vector<SystemRunData*> runs;
    boost::mutex runMutex;
    boost::condition_variable runCondition;
};

// Define template functions here to avoid linker errors
template <typename T>
T* SystemRunData::getData(const std::string key) {
    boost::unique_lock<boost::mutex> lock(this->dataMutex);
    if (!this->data.count(key)) {
        throw std::invalid_argument("Could not find key");
    }

    return static_cast<T*>(this->data[key]);
}

template <typename T>
boost::future<T*> SystemRunData::getDataAsync(const std::string key) {
    boost::packaged_task<T*> task([this, key] {
        boost::unique_lock<boost::mutex> lock(this->dataMutex);
        while (!this->data.count(key)) {
            LOG4CXX_DEBUG(this->system->logger, "Waiting for key " << key << " to be available");
            this->dataCondition.wait(lock);
        }

        LOG4CXX_DEBUG(this->system->logger, "Key " << key << " is now available");
        return static_cast<T*>(this->data[key]);
    });

    auto future = task.get_future();
    boost::asio::post(this->system->threadPool, boost::move(task));
    return future;
}
}  // namespace cart

#endif  // CARTSLAM_HPP