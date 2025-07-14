// copyright: gpt o1-preview-0912-global

#include <string>
#include <random>

namespace rtp_llm {
namespace rtp_llm_master {

class RandomStringGenerator {
public:
    RandomStringGenerator():
        charset("0123456789"
                "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        distribution(0, charset.size() - 1),
        generator(std::random_device{}()) {}

    std::string getRandomString(std::size_t length = 16) {
        std::string random_string;
        random_string.reserve(length);
        for (std::size_t i = 0; i < length; ++i) {
            random_string += charset[distribution(generator)];
        }
        return random_string;
    }

private:
    const std::string               charset;
    std::uniform_int_distribution<> distribution;
    std::mt19937                    generator;
};

}  // namespace rtp_llm_master
}  // namespace rtp_llm
