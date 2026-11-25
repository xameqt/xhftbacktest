#ifndef XHFTBACKTEST_MMAP_LOADER_HPP
#define XHFTBACKTEST_MMAP_LOADER_HPP

#include "event.hpp"

#include <bit>
#include <cerrno>
#include <charconv>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace xhftbacktest {

// A borrowed span stays valid while its mapping is owned by a loader.
class MmapLoader {
    using View = std::string_view;
    void* mapping_ = MAP_FAILED;
    std::size_t bytes_ = 0;
    std::span<const Event> events_;

    struct Header {
        View rest;

        static void require(bool ok) {
            if (!ok) throw std::runtime_error("invalid or incompatible Event .npy header");
        }
        void space() {
            const auto i = rest.find_first_not_of(" \t\r\n");
            rest.remove_prefix(i == View::npos ? rest.size() : i);
        }
        bool take(View token) {
            space();
            if (!rest.starts_with(token)) return false;
            rest.remove_prefix(token.size());
            return true;
        }
        View quoted() {
            space();
            require(rest.starts_with('\'') || rest.starts_with('"'));
            const char quote = rest.front();
            rest.remove_prefix(1);
            const auto end = rest.find(quote);
            require(end != View::npos);
            const View value = rest.substr(0, end);
            rest.remove_prefix(end + 1);
            return value;
        }
        void descriptor() {
            constexpr View names[] = {"ev", "exch_ts", "local_ts", "px", "qty", "order_id", "ival", "fval"};
            constexpr View kinds = "uiiffuif";
            constexpr char endian = std::endian::native == std::endian::little ? '<' : '>';
            require(take("["));
            for (std::size_t i = 0; i < kinds.size(); ++i) {
                if (i) require(take(","));
                require(take("(") && quoted() == names[i] && take(","));
                View fmt = quoted();
                if (fmt.starts_with(endian) || fmt.starts_with('=')) fmt.remove_prefix(1);
                require(fmt.size() == 2 && fmt[0] == kinds[i] && fmt[1] == '8' && take(")"));
            }
            take(",");
            require(take("]"));
        }
        std::size_t shape() {
            require(take("("));
            space();
            std::size_t n = 0;
            const auto [end, err] = std::from_chars(rest.data(), rest.data() + rest.size(), n);
            require(err == std::errc{});
            rest.remove_prefix(static_cast<std::size_t>(end - rest.data()));
            require(take(",") && take(")"));
            return n;
        }
        std::size_t parse() {
            require(take("{"));
            bool dtype = false, order = false, length = false;
            std::size_t n = 0;
            while (!take("}")) {
                const View key = quoted();
                require(take(":"));
                if (key == "descr" && !dtype) {
                    descriptor();
                    dtype = true;
                } else if (key == "fortran_order" && !order) {
                    require(take("False") || take("True")); // Identical for 1-D arrays.
                    order = true;
                } else if (key == "shape" && !length) {
                    n = shape();
                    length = true;
                } else {
                    require(false);
                }
                if (!take(",")) {
                    require(take("}"));
                    break;
                }
            }
            space();
            require(dtype && order && length && rest.empty());
            return n;
        }
    };

    static std::size_t npy_offset(View file) {
        Header::require(file.size() >= 10 && file.starts_with("\x93NUMPY") && file[7] == 0);
        const int version = file[6];
        Header::require(version >= 1 && version <= 3);
        const std::size_t prefix = version == 1 ? 10 : 12;
        Header::require(file.size() >= prefix);
        std::size_t length = 0;
        for (std::size_t i = 8; i < prefix; ++i)
            length |= std::size_t(static_cast<unsigned char>(file[i])) << (8 * (i - 8));
        Header::require(length <= file.size() - prefix);
        const std::size_t offset = prefix + length;
        const auto header = file.substr(prefix, length);
        Header::require(offset % alignof(Event) == 0 && header.ends_with('\n'));
        const auto count = Header{header}.parse();
        const auto bytes = file.size() - offset;
        Header::require(bytes % sizeof(Event) == 0 && count == bytes / sizeof(Event));
        return offset;
    }

    void swap(MmapLoader& other) noexcept {
        std::swap(mapping_, other.mapping_);
        std::swap(bytes_, other.bytes_);
        std::swap(events_, other.events_);
    }

public:
    MmapLoader() noexcept = default;

    explicit MmapLoader(const std::filesystem::path& path) : MmapLoader() {
        const bool npy = path.extension() == ".npy";
        if (!npy && path.extension() != ".bin")
            throw std::invalid_argument("expected an Event .bin or .npy file");
        struct File {
            int fd;
            ~File() { if (fd >= 0) ::close(fd); }
        } file{::open(path.c_str(), O_RDONLY | O_CLOEXEC)};
        if (file.fd < 0) throw std::system_error(errno, std::generic_category(), "open");
        struct stat info{};
        if (::fstat(file.fd, &info) < 0) throw std::system_error(errno, std::generic_category(), "fstat");
        if (!S_ISREG(info.st_mode) || !std::in_range<std::size_t>(info.st_size))
            throw std::runtime_error("expected a regular file of addressable size");
        bytes_ = static_cast<std::size_t>(info.st_size);
        if (!npy && bytes_ % sizeof(Event)) throw std::runtime_error("partial Event in .bin file");
        if (bytes_ == 0) {
            if (npy) throw std::runtime_error("missing .npy header");
            return;
        }
        mapping_ = ::mmap(nullptr, bytes_, PROT_READ, MAP_PRIVATE, file.fd, 0);
        if (mapping_ == MAP_FAILED) throw std::system_error(errno, std::generic_category(), "mmap");
        const auto* data = static_cast<const char*>(mapping_);
        const auto offset = npy ? npy_offset({data, bytes_}) : 0;
        events_ = {reinterpret_cast<const Event*>(data + offset), (bytes_ - offset) / sizeof(Event)};
    }

    ~MmapLoader() { if (mapping_ != MAP_FAILED) ::munmap(mapping_, bytes_); }
    MmapLoader(const MmapLoader&) = delete;
    MmapLoader& operator=(const MmapLoader&) = delete;
    MmapLoader(MmapLoader&& other) noexcept { swap(other); }
    MmapLoader& operator=(MmapLoader&& other) noexcept {
        MmapLoader incoming(std::move(other));
        swap(incoming);
        return *this;
    }

    [[nodiscard]] std::span<const Event> events() const noexcept { return events_; }
};

} // namespace xhftbacktest
#endif
