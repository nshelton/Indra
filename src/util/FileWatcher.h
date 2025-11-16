#pragma once

#include <string>
#include <vector>
#include <functional>
#include <filesystem>
#include <unordered_map>

// Simple file watcher that polls for file changes
// Uses modification time checking rather than OS-specific file notifications
class FileWatcher
{
public:
    using CallbackFunc = std::function<void(const std::string& path)>;

    FileWatcher();

    // Add a file to watch with a callback
    void watch(const std::string& path, CallbackFunc callback);

    // Add all files in a directory to watch with a callback
    // If recursive is true, watches subdirectories as well
    void watchDirectory(const std::string& directory, CallbackFunc callback, bool recursive = false);

    // Remove a file from watching
    void unwatch(const std::string& path);

    // Check all watched files for changes (call this in your update loop)
    void update();

    // Clear all watched files
    void clear();

private:
    struct WatchedFile
    {
        std::string path;
        std::filesystem::file_time_type lastModTime;
        CallbackFunc callback;
    };

    std::unordered_map<std::string, WatchedFile> m_watchedFiles;

    std::filesystem::file_time_type getFileModTime(const std::string& path);
};
