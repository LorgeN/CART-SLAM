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
#define MODULE_RETURN(key, value) std::make_optional(std::make_pair(key, value))
#define MODULE_RETURN_VALUE_PAIR std::pair<std::string, boost::shared_ptr<void>>
#define MODULE_RETURN_VALUE std::optional<MODULE_RETURN_VALUE_PAIR>

#include <log4cxx/logger.h>

#include <boost/asio/post.hpp>
#include <boost/asio/thread_pool.hpp>
#include <boost/enable_shared_from_this.hpp>
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
    SystemRunData(uint32_t id, boost::weak_ptr<System> system, boost::shared_ptr<DataElement> dataElement) : id(id), system(system), dataElement(dataElement) {}
    ~SystemRunData() = default;

    template <typename T>
    boost::shared_ptr<T> getData(const std::string key);

    template <typename T>
    boost::future<boost::shared_ptr<T>> getDataAsync(const std::string key);

    void insertData(MODULE_RETURN_VALUE_PAIR data);

    void markAsComplete();
    bool isComplete();

    boost::shared_ptr<SystemRunData> getRelativeRun(const int8_t offset);

    boost::shared_ptr<DataElement> dataElement;
    const uint32_t id;

    log4cxx::LoggerPtr getLogger();
    boost::asio::thread_pool& getThreadPool();

   private:
    bool complete = false;
    boost::weak_ptr<System> system;
    std::map<std::string, boost::shared_ptr<void>> data;
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

class System : public boost::enable_shared_from_this<System> {
   public:
    System(boost::shared_ptr<DataSource> dataSource);
    ~System() = default;
    boost::future<void> run();
    void addModule(boost::shared_ptr<SystemModule> module);

    boost::shared_ptr<SystemRunData> startNewRun(cv::cuda::Stream& stream);
    boost::shared_ptr<SystemRunData> getRunById(const uint32_t index);
    uint8_t getActiveRunCount();

    log4cxx::LoggerPtr logger;
    boost::asio::thread_pool threadPool = boost::asio::thread_pool(CARTSLAM_WORKER_THREADS);

   private:
    uint32_t runId = 0;
    boost::shared_ptr<DataSource> dataSource;
    std::vector<boost::shared_ptr<SystemModule>> modules;
    std::vector<boost::shared_ptr<SystemRunData>> runs;
    boost::mutex runMutex;
    boost::condition_variable runCondition;
};

// Define template functions here to avoid linker errors
template <typename T>
boost::shared_ptr<T> SystemRunData::getData(const std::string key) {
    boost::unique_lock<boost::mutex> lock(this->dataMutex);
    if (!this->data.count(key)) {
        throw std::invalid_argument("Could not find key");
    }

    return boost::static_pointer_cast<T>(this->data[key]);
}

template <typename T>
boost::future<boost::shared_ptr<T>> SystemRunData::getDataAsync(const std::string key) {
    boost::packaged_task<boost::shared_ptr<T>> task([this, key] {
        boost::unique_lock<boost::mutex> lock(this->dataMutex);
        while (!this->data.count(key)) {
            LOG4CXX_DEBUG(this->getLogger(), "Waiting for key " << key << " to be available");
            this->dataCondition.wait(lock);
        }

        LOG4CXX_DEBUG(this->getLogger(), "Key " << key << " is now available");
        return boost::static_pointer_cast<T>(this->data[key]);
    });

    auto future = task.get_future();
    boost::asio::post(this->getThreadPool(), boost::move(task));
    return future;
}
}  // namespace cart

#endif  // CARTSLAM_HPP