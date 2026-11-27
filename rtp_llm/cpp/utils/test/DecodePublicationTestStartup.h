#pragma once

#include <cstdlib>
#include <iostream>
#include <string>

#include <gtest/gtest.h>

namespace rtp_llm::test {

// The production switch is cached at first use. A fresh test process
// exercises each environment as a separate service startup would.
template<typename Run>
void withDecodePublicationStartup(Run&& run) {
    struct RestoreStyle {
        std::string previous = ::testing::GTEST_FLAG(death_test_style);
        ~RestoreStyle() {
            ::testing::GTEST_FLAG(death_test_style) = previous;
        }
    } restore;
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    ASSERT_EXIT(
        {
            run();
            const auto& result = *::testing::UnitTest::GetInstance()->current_test_info()->result();
            for (int i = 0; i < result.total_part_count(); ++i) {
                const auto& part = result.GetTestPartResult(i);
                if (part.failed()) {
                    std::cerr << part.file_name() << ':' << part.line_number() << ' ' << part.message() << '\n';
                }
            }
            std::_Exit(result.Failed() ? EXIT_FAILURE : EXIT_SUCCESS);
        },
        ::testing::ExitedWithCode(EXIT_SUCCESS), "");
}

// Fixture-owned resources need teardown inside the exec child before _Exit.
template<typename Run, typename Cleanup>
void withDecodePublicationStartup(Run&& run, Cleanup&& cleanup) {
    withDecodePublicationStartup([&] {
        run();
        cleanup();
    });
}

}  // namespace rtp_llm::test
