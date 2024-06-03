#pragma once

#ifdef CARTSLAM_IMAGE_MAKE_GRAYSCALE
#define CARTSLAM_RUN_RETENTION 32
#define CARTSLAM_CONCURRENT_RUN_LIMIT 12
#define CARTSLAM_IMAGE_CHANNELS 1
#else
#define CARTSLAM_RUN_RETENTION 24
#define CARTSLAM_CONCURRENT_RUN_LIMIT 8
#define CARTSLAM_IMAGE_CHANNELS 3
#endif

#define CARTSLAM_WORKER_THREADS (16 * CARTSLAM_CONCURRENT_RUN_LIMIT)

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
#include "modules/module.hpp"
#include "opencv2/core/cuda.hpp"
#include "utils/data.hpp"

namespace cart {

class System;  // Allow references

class SystemRunData : public DataContainer {
   public:
    SystemRunData(uint32_t id, boost::weak_ptr<System> system, boost::shared_ptr<DataElement> dataElement) : id(id), system(system), dataElement(dataElement) {
        this->logger = cart::getLogger("Run " + std::to_string(id));
    }

    ~SystemRunData() = default;

    void markAsComplete();
    bool isComplete();

    boost::shared_ptr<SystemRunData> getRelativeRun(const int8_t offset);

    boost::shared_ptr<DataElement> dataElement;
    const uint32_t id;

    log4cxx::LoggerPtr getLogger() override;
    boost::asio::thread_pool& getThreadPool() override;

    friend class System;

   private:
    bool complete = false;
    boost::weak_ptr<System> system;
    log4cxx::LoggerPtr logger;
};

class System : public boost::enable_shared_from_this<System>, public DataContainer {
   public:
    System(boost::shared_ptr<DataSource> dataSource);
    ~System() = default;
    boost::future<void> run();

    template <typename T, typename... Args>
    void addModule(Args... args) {
        this->addModule(boost::make_shared<T>(args...));
    }

    void addModule(boost::shared_ptr<SystemModule> module);

    template <typename T>
    boost::shared_ptr<T> getModule() {
        static_assert(std::is_base_of<SystemModule, T>::value, "T must be a subclass of SystemModule");

        for (const auto& module : this->modules) {
            auto casted = boost::dynamic_pointer_cast<T>(module);
            if (casted) {
                return casted;
            }
        }

        throw std::invalid_argument("Could not find module");
    }

    boost::shared_ptr<SystemRunData> startNewRun(cv::cuda::Stream& stream);
    boost::shared_ptr<SystemRunData> getRunById(const uint32_t index);
    uint8_t getActiveRunCount();

    log4cxx::LoggerPtr getLogger() override;
    boost::asio::thread_pool& getThreadPool() override;

    void insertGlobalData(const std::string key, boost::shared_ptr<void> data);
    void insertGlobalData(system_data_pair_t data);

    const boost::shared_ptr<const DataSource> getDataSource() const;

   private:
    void verifyDependencies();

    boost::future<void> waitForDependencies(const std::vector<module_dependency_t>& dependencies, boost::shared_ptr<SystemRunData> data);

    bool verifiedDependencies = false;

    log4cxx::LoggerPtr logger;
    boost::asio::thread_pool threadPool = boost::asio::thread_pool(CARTSLAM_WORKER_THREADS);

    uint32_t runId = 0;
    boost::shared_ptr<DataSource> dataSource;
    std::vector<boost::shared_ptr<SystemModule>> modules;
    std::map<std::string, boost::shared_ptr<SystemModule>> dataProvidedBy;
    std::vector<boost::shared_ptr<SystemRunData>> runs;
    boost::shared_mutex runMutex;
    boost::condition_variable_any runCondition;
};
}  // namespace cart
