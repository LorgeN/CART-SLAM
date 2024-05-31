#include "utils/data.hpp"

#include <boost/throw_exception.hpp>

namespace cart {

const std::vector<std::string> DataContainer::getDataKeys() {
    boost::lock_guard<boost::mutex> lock(this->dataMutex);
    std::vector<std::string> keys;
    for (const auto& pair : this->data) {
        keys.push_back(pair.first);
    }

    return keys;
}

void DataContainer::insertData(system_data_pair_t data) {
    LOG4CXX_DEBUG(this->getLogger(), "Inserting data with key " << std::quoted(data.first));

    {
        boost::lock_guard<boost::mutex> lock(this->dataMutex);
        this->data[data.first] = data.second;
        LOG4CXX_DEBUG(this->getLogger(), "Inserted data with key " << std::quoted(data.first));
    }

    this->dataCondition.notify_all();
}

bool DataContainer::hasData(const std::string key) {
    boost::lock_guard<boost::mutex> lock(this->dataMutex);
    return this->data.count(key);
}

boost::future<void> DataContainer::waitForData(const std::vector<std::string> keys) {
    if (keys.empty()) {
        throw std::invalid_argument("No keys provided to wait for");
    }

    boost::packaged_task<void> task([this, keys] {
        for (const auto& key : keys) {
            boost::unique_lock<boost::mutex> lock(this->dataMutex);
            // If we ever have to wait more than a few seconds for new data to be inserted, the run
            // is most likely over and something has failed somewhere
            const boost::system_time timeout = boost::get_system_time() + boost::posix_time::seconds(CARTSLAM_WAIT_FOR_DATA_TIMEOUT);

            if (!this->dataCondition.timed_wait(lock, timeout, [this, key] { return this->data.count(key); })) {
                LOG4CXX_ERROR(this->getLogger(), "Timed out while waiting for " << std::quoted(key));
                BOOST_THROW_EXCEPTION(DataNotAvailableException(key));
            }
        }
    });

    auto future = task.get_future();
    boost::asio::post(this->getThreadPool(), boost::move(task));
    return future;
}
}  // namespace cart