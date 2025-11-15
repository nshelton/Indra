#include "FileWatcher.h"
#include <filesystem>
#include <glog/logging.h>

FileWatcher::FileWatcher()
{
}

std::filesystem::file_time_type FileWatcher::getFileModTime(const std::string& path)
{
    try
    {
        return std::filesystem::last_write_time(path);
    }
    catch (const std::filesystem::filesystem_error& e)
    {
        // File might not exist or be inaccessible
        return std::filesystem::file_time_type{};
    }
}

void FileWatcher::watch(const std::string& path, CallbackFunc callback)
{
    auto modTime = getFileModTime(path);

    WatchedFile wf;
    wf.path = path;
    wf.lastModTime = modTime;
    wf.callback = callback;

    m_watchedFiles[path] = wf;

    LOG(INFO) << "FileWatcher: Now watching " << path;
}

void FileWatcher::unwatch(const std::string& path)
{
    auto it = m_watchedFiles.find(path);
    if (it != m_watchedFiles.end())
    {
        m_watchedFiles.erase(it);
        LOG(INFO) << "FileWatcher: Stopped watching " << path;
    }
}

void FileWatcher::update()
{
    for (auto& [path, watchedFile] : m_watchedFiles)
    {
        auto currentModTime = getFileModTime(watchedFile.path);

        // Check if file was modified
        if (currentModTime > watchedFile.lastModTime && currentModTime.time_since_epoch().count() > 0)
        {
            LOG(INFO) << "FileWatcher: Detected change in " << watchedFile.path;

            // Update the modification time
            watchedFile.lastModTime = currentModTime;

            // Call the callback
            if (watchedFile.callback)
            {
                watchedFile.callback(watchedFile.path);
            }
        }
    }
}

void FileWatcher::clear()
{
    m_watchedFiles.clear();
    LOG(INFO) << "FileWatcher: Cleared all watched files";
}
