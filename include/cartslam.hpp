#pragma once

#define BOOST_THREAD_VERSION 3
#define BOOST_THREAD_PROVIDES_FUTURE
#define BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION
#define BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY
#define BOOST_THREAD_PROVIDES_FUTURE_UNWRAP

#define CARTSLAM_WORKER_THREADS 16

#define MODULE_NO_RETURN_VALUE std::nullopt
#define MODULE_RETURN(key, value) std::make_optional(std::make_pair(key, value))

#ifdef CARTSLAM_IMAGE_MAKE_GRAYSCALE
#define CARTSLAM_RUN_RETENTION 10
#define CARTSLAM_CONCURRENT_RUN_LIMIT 6
#define CARTSLAM_IMAGE_CHANNELS 1
#else
#define CARTSLAM_RUN_RETENTION 6
#define CARTSLAM_CONCURRENT_RUN_LIMIT 2
#define CARTSLAM_IMAGE_CHANNELS 3
#endif

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

typedef std::pair<std::string, boost::shared_ptr<void>> module_result_pair_t;
typedef std::optional<module_result_pair_t> module_result_t;

class System;  // Allow references

class SystemRunData {
   public:
    SystemRunData(uint32_t id, boost::weak_ptr<System> system, boost::shared_ptr<DataElement> dataElement) : id(id), system(system), dataElement(dataElement) {}
    ~SystemRunData() = default;

    template <typename T>
    boost::shared_ptr<T> getData(const std::string key) {
        boost::unique_lock<boost::mutex> lock(this->dataMutex);
        if (!this->data.count(key)) {
            throw std::invalid_argument("Could not find key");
        }

        return boost::static_pointer_cast<T>(this->data[key]);
    }

    template <typename T>
    boost::future<boost::shared_ptr<T>> getDataAsync(const std::string key) {
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

    boost::future<void> waitForData(const std::vector<std::string> keys) {
        boost::packaged_task<void> task([this, keys] {
            boost::unique_lock<boost::mutex> lock(this->dataMutex);
            for (const auto& key : keys) {
                LOG4CXX_DEBUG(this->getLogger(), "Waiting for key " << key << " to be available");

                while (!this->data.count(key)) {
                    // If we ever have to wait more than 3 seconds for new data to be inserted, the run
                    // is most likely over and something has failed somewhere
                    const boost::system_time timeout = boost::get_system_time() + boost::posix_time::seconds(3);

                    if (!this->dataCondition.timed_wait(lock, timeout)) {
                        throw std::runtime_error("Timeout waiting for data key \"" + key + "\"");
                    }
                }
            }
        });

        auto future = task.get_future();
        boost::asio::post(this->getThreadPool(), boost::move(task));
        return future;
    }

    void markAsComplete();
    bool isComplete();

    boost::shared_ptr<SystemRunData> getRelativeRun(const int8_t offset);

    boost::shared_ptr<DataElement> dataElement;
    const uint32_t id;

    log4cxx::LoggerPtr getLogger();
    boost::asio::thread_pool& getThreadPool();

    friend class System;

   private:
    void insertData(module_result_pair_t data);

    bool complete = false;
    boost::weak_ptr<System> system;
    std::map<std::string, boost::shared_ptr<void>> data;
    boost::mutex dataMutex;
    boost::condition_variable dataCondition;
};

class SystemModule {
   public:
    SystemModule(const std::string& name) : SystemModule(name, {}){};

    SystemModule(const std::string& name, const std::vector<std::string> requiresData) : name(name), requiresData(requiresData) {
        this->logger = getLogger(name);
    }

    virtual ~SystemModule() = default;
    virtual boost::future<module_result_t> run(System& system, SystemRunData& data) = 0;

    const std::vector<std::string> requiresData;

    // TODO: Add check for missing data dependencies
    // const std::string providesData;

    const std::string name;

   protected:
    log4cxx::LoggerPtr logger;
};

class SyncWrapperSystemModule : public SystemModule {
   public:
    SyncWrapperSystemModule(const std::string& name) : SystemModule(name){};

    SyncWrapperSystemModule(const std::string& name, const std::vector<std::string> requiresData) : SystemModule(name, requiresData){};

    boost::future<module_result_t> run(System& system, SystemRunData& data) override;
    virtual module_result_t runInternal(System& system, SystemRunData& data) = 0;
};

class System : public boost::enable_shared_from_this<System> {
   public:
    System(boost::shared_ptr<DataSource> dataSource);
    ~System() = default;
    boost::future<void> run();

    template <typename T, typename... Args>
    void addModule(Args... args) {
        this->addModule(boost::make_shared<T>(args...));
    }

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
}  // namespace cart
