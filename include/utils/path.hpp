#pragma once

#include <string>

namespace cart::util {
inline const std::string resolvePath(const std::string& path) {
    // Replace ~ with the home directory
    if (path[0] == '~') {
        const char* home = getenv("HOME");
        if (home) {
            return std::string(home) + path.substr(1);
        }
    }

    return path;
}
}  // namespace cart::util