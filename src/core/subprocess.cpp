#include "subprocess.hpp"

#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstring>
#include <chrono>
#include <thread>
#include <mutex>
#include <signal.h>

extern char** environ;

namespace derep {

namespace {

std::string read_fd(int fd) {
    std::string result;
    char buf[4096];
    for (;;) {
        ssize_t n = ::read(fd, buf, sizeof(buf));
        if (n > 0) {
            result.append(buf, static_cast<std::size_t>(n));
        } else if (n == 0) {
            break;
        } else {
            if (errno == EINTR) continue;
            break;
        }
    }
    return result;
}

// posix_spawn_file_actions_addchdir_np is available on glibc >= 2.29 (Linux 5.5+).
// On older systems we fall back to chdir+restore protected by a mutex.
#if defined(__GLIBC__) && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 29))
#define HAVE_SPAWN_CHDIR 1
#else
#define HAVE_SPAWN_CHDIR 0
#endif

#if !HAVE_SPAWN_CHDIR
std::mutex g_chdir_mutex;
#endif

} // namespace

SubprocessResult run_subprocess(
    const std::vector<std::string>& argv,
    const SubprocessOptions& options)
{
    if (argv.empty()) {
        throw SubprocessError("argv must not be empty");
    }

    // Build C argv
    std::vector<char*> argv_cstrs;
    argv_cstrs.reserve(argv.size() + 1);
    for (const auto& arg : argv) {
        argv_cstrs.push_back(const_cast<char*>(arg.c_str()));
    }
    argv_cstrs.push_back(nullptr);

    // Build envp
    std::vector<std::string> env_storage;
    std::vector<char*> envp;

    if (options.clear_environment) {
        for (const auto& [key, val] : options.env_overrides) {
            env_storage.push_back(key + "=" + val);
        }
    } else {
        // Copy current environment
        std::unordered_map<std::string, std::string> merged;
        for (char** e = environ; e && *e; ++e) {
            std::string entry(*e);
            auto eq = entry.find('=');
            if (eq != std::string::npos) {
                merged[entry.substr(0, eq)] = entry.substr(eq + 1);
            }
        }
        // Apply overrides
        for (const auto& [key, val] : options.env_overrides) {
            merged[key] = val;
        }
        env_storage.reserve(merged.size());
        for (const auto& [key, val] : merged) {
            env_storage.push_back(key + "=" + val);
        }
    }

    envp.reserve(env_storage.size() + 1);
    for (auto& s : env_storage) {
        envp.push_back(s.data());
    }
    envp.push_back(nullptr);

    // Set up file actions
    posix_spawn_file_actions_t file_actions;
    posix_spawn_file_actions_init(&file_actions);

    int stdout_pipe[2] = {-1, -1};
    int stderr_pipe[2] = {-1, -1};

    if (options.stdout_file) {
        posix_spawn_file_actions_addopen(
            &file_actions, STDOUT_FILENO,
            options.stdout_file->c_str(),
            O_WRONLY | O_CREAT | O_TRUNC, 0644);
    } else if (options.capture_stdout) {
        if (::pipe(stdout_pipe) != 0) {
            posix_spawn_file_actions_destroy(&file_actions);
            throw SubprocessError("pipe() for stdout failed: " + std::string(std::strerror(errno)));
        }
        posix_spawn_file_actions_adddup2(&file_actions, stdout_pipe[1], STDOUT_FILENO);
        posix_spawn_file_actions_addclose(&file_actions, stdout_pipe[0]);
    }

    if (options.capture_stderr) {
        if (::pipe(stderr_pipe) != 0) {
            if (stdout_pipe[0] >= 0) { ::close(stdout_pipe[0]); ::close(stdout_pipe[1]); }
            posix_spawn_file_actions_destroy(&file_actions);
            throw SubprocessError("pipe() for stderr failed: " + std::string(std::strerror(errno)));
        }
        posix_spawn_file_actions_adddup2(&file_actions, stderr_pipe[1], STDERR_FILENO);
        posix_spawn_file_actions_addclose(&file_actions, stderr_pipe[0]);
    }

    // Working directory
#if HAVE_SPAWN_CHDIR
    if (options.working_dir) {
        posix_spawn_file_actions_addchdir_np(&file_actions, options.working_dir->c_str());
    }
#endif

    auto start = std::chrono::steady_clock::now();

    pid_t pid = -1;
    int spawn_ret;

#if HAVE_SPAWN_CHDIR
    spawn_ret = posix_spawnp(&pid, argv[0].c_str(), &file_actions, nullptr,
                             argv_cstrs.data(), envp.data());
#else
    if (options.working_dir) {
        // Fallback: chdir in parent, spawn, restore. Mutex-protected to avoid races.
        std::lock_guard<std::mutex> lock(g_chdir_mutex);
        char old_cwd[4096];
        if (!::getcwd(old_cwd, sizeof(old_cwd))) {
            posix_spawn_file_actions_destroy(&file_actions);
            if (stdout_pipe[0] >= 0) { ::close(stdout_pipe[0]); ::close(stdout_pipe[1]); }
            if (stderr_pipe[0] >= 0) { ::close(stderr_pipe[0]); ::close(stderr_pipe[1]); }
            throw SubprocessError("getcwd failed: " + std::string(std::strerror(errno)));
        }
        if (::chdir(options.working_dir->c_str()) != 0) {
            posix_spawn_file_actions_destroy(&file_actions);
            if (stdout_pipe[0] >= 0) { ::close(stdout_pipe[0]); ::close(stdout_pipe[1]); }
            if (stderr_pipe[0] >= 0) { ::close(stderr_pipe[0]); ::close(stderr_pipe[1]); }
            throw SubprocessError("chdir to working_dir failed: " + std::string(std::strerror(errno)));
        }
        spawn_ret = posix_spawnp(&pid, argv[0].c_str(), &file_actions, nullptr,
                                 argv_cstrs.data(), envp.data());
        ::chdir(old_cwd);
    } else {
        spawn_ret = posix_spawnp(&pid, argv[0].c_str(), &file_actions, nullptr,
                                 argv_cstrs.data(), envp.data());
    }
#endif

    posix_spawn_file_actions_destroy(&file_actions);

    if (spawn_ret != 0) {
        if (stdout_pipe[0] >= 0) { ::close(stdout_pipe[0]); ::close(stdout_pipe[1]); }
        if (stderr_pipe[0] >= 0) { ::close(stderr_pipe[0]); ::close(stderr_pipe[1]); }
        throw SubprocessError("posix_spawnp failed: " + std::string(std::strerror(spawn_ret)));
    }

    // Close write ends in parent
    if (stdout_pipe[1] >= 0) ::close(stdout_pipe[1]);
    if (stderr_pipe[1] >= 0) ::close(stderr_pipe[1]);

    // Read pipe outputs
    SubprocessResult result;

    if (stdout_pipe[0] >= 0) {
        result.stdout_output = read_fd(stdout_pipe[0]);
        ::close(stdout_pipe[0]);
    }
    if (stderr_pipe[0] >= 0) {
        result.stderr_output = read_fd(stderr_pipe[0]);
        ::close(stderr_pipe[0]);
    }

    // Wait with timeout
    auto deadline = start + options.timeout;
    int status = 0;
    bool exited = false;

    while (!exited) {
        int wr = ::waitpid(pid, &status, WNOHANG);
        if (wr == pid) {
            exited = true;
            break;
        }
        if (wr < 0 && errno != EINTR) {
            break;
        }

        if (std::chrono::steady_clock::now() >= deadline) {
            result.timed_out = true;
            ::kill(pid, SIGTERM);
            std::this_thread::sleep_for(std::chrono::seconds(5));
            // Check if it exited after SIGTERM
            wr = ::waitpid(pid, &status, WNOHANG);
            if (wr != pid) {
                ::kill(pid, SIGKILL);
                ::waitpid(pid, &status, 0);
            }
            exited = true;
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    auto end = std::chrono::steady_clock::now();
    result.wall_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (WIFEXITED(status)) {
        result.exit_code = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        result.exit_code = 128 + WTERMSIG(status);
    }

    return result;
}

} // namespace derep
