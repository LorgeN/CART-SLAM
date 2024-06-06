#pragma once

// From https://stackoverflow.com/questions/25201131/writing-csv-files-from-c

#include <fstream>
#include <iostream>
#include <string>

namespace cart::utils {
class csvfile;

inline static csvfile& endrow(csvfile& file);
inline static csvfile& flush(csvfile& file);

class csvfile {
   private:
    std::ofstream file;
    const std::string separator;

   public:
    csvfile(const std::string filename, const std::string separator = ";", const std::vector<std::string> headers = {})
        : file(), separator(separator) {
        file.exceptions(std::ios::failbit | std::ios::badbit);
        file.open(filename);

        if (!headers.empty()) {
            for (const auto& header : headers) {
                file << header << separator;
            }
            
            file << std::endl;
        }
    }

    ~csvfile() {
        flush();
        file.close();
    }

    void flush() {
        file.flush();
    }

    void endrow() {
        file << std::endl;
    }

    csvfile& operator<<(csvfile& (*val)(csvfile&)) {
        return val(*this);
    }

    csvfile& operator<<(const char* val) {
        file << '"' << val << '"' << separator;
        return *this;
    }

    csvfile& operator<<(const std::string& val) {
        file << '"' << val << '"' << separator;
        return *this;
    }

    template <typename T>
    csvfile& operator<<(const T& val) {
        file << val << separator;
        return *this;
    }
};

inline static csvfile& endrow(csvfile& file) {
    file.endrow();
    return file;
}

inline static csvfile& flush(csvfile& file) {
    file.flush();
    return file;
}
}  // namespace cart::utils