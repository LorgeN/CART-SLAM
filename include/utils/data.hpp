#pragma once

#include <log4cxx/logger.h>

#include <boost/asio/post.hpp>
#include <boost/asio/thread_pool.hpp>
#include <boost/thread.hpp>
#include <boost/thread/future.hpp>
#include <vector>

#define CARTSLAM_WAIT_FOR_DATA_TIMEOUT 20

namespace cart {
typedef std::pair<std::string, boost::shared_ptr<void>> system_data_pair_t;
typedef std::vector<system_data_pair_t> system_data_t;

class DataContainer {
   public:
    DataContainer() = default;

    virtual ~DataContainer() = default;

    virtual log4cxx::LoggerPtr getLogger() = 0;

    virtual boost::asio::thread_pool& getThreadPool() = 0;

    bool hasData(const std::string key);

    const std::vector<std::string> getDataKeys();

    template <typename T>
    boost::shared_ptr<T> getData(const std::string key) {
        boost::unique_lock<boost::mutex> lock(this->dataMutex);
        if (!this->data.count(key)) {
            throw std::invalid_argument("Could not find key \"" + key + "\"");
        }

        return boost::static_pointer_cast<T>(this->data[key]);
    }

    template <typename T>
    boost::future<boost::shared_ptr<T>> getDataAsync(const std::string key) {
        return this->waitForData({key}).then([this, key](boost::future<void> future) {
            try {
                future.get();
            } catch (const boost::exception& e) {
                LOG4CXX_ERROR(this->getLogger(), "Error while waiting for data: " << boost::diagnostic_information(e));
                throw;
            }

            LOG4CXX_DEBUG(this->getLogger(), "Data key " << key << " is now available");
            return this->getData<T>(key);
        });
    }

    boost::future<void> waitForData(const std::vector<std::string> keys);

   protected:
    void insertData(system_data_pair_t data);

   private:
    std::map<std::string, boost::shared_ptr<void>> data;
    boost::mutex dataMutex;
    boost::condition_variable dataCondition;
};
}  // namespace cart