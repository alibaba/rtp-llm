cmake_minimum_required(VERSION 3.16)
project(block_tree_cache VERSION 2.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CTest)
find_package(Threads REQUIRED)

# These are the same workflow translation units used by the RTP coordinator.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS native_sources.bzl)
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/native_sources.bzl" BTC_SOURCE_LINES REGEX "[.]cc\",")
set(BTC_NATIVE_SOURCES)
foreach(line IN LISTS BTC_SOURCE_LINES)
    string(REGEX REPLACE "^ *\"([^\"]+)\",.*" "\\1" source "${line}")
    list(APPEND BTC_NATIVE_SOURCES "${source}")
endforeach()

add_library(block_tree_cache_native OBJECT ${BTC_NATIVE_SOURCES})
get_filename_component(BTC_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/support/rtp_llm")
    # A standalone sdist contains the small, standard-library-only support headers.
    set(BTC_REPO_ROOT "${CMAKE_CURRENT_BINARY_DIR}/native_include")
    file(MAKE_DIRECTORY "${BTC_REPO_ROOT}/rtp_llm/cpp/cache")
    file(CREATE_LINK "${CMAKE_CURRENT_SOURCE_DIR}" "${BTC_REPO_ROOT}/rtp_llm/cpp/cache/block_tree_cache" SYMBOLIC)
endif()
target_include_directories(block_tree_cache_native PRIVATE . include "${BTC_REPO_ROOT}" "${CMAKE_CURRENT_SOURCE_DIR}/support")
target_compile_features(block_tree_cache_native PRIVATE cxx_std_17)
target_link_libraries(block_tree_cache_native PRIVATE Threads::Threads)
set_target_properties(block_tree_cache_native PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(block_tree_cache SHARED $<TARGET_OBJECTS:block_tree_cache_native>)
target_compile_features(block_tree_cache PUBLIC cxx_std_17)
target_include_directories(block_tree_cache
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>" "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    PRIVATE . "${BTC_REPO_ROOT}")
target_link_libraries(block_tree_cache PRIVATE Threads::Threads)
target_compile_definitions(block_tree_cache PUBLIC BTC_SHARED PRIVATE BTC_BUILDING_LIBRARY)
set_target_properties(block_tree_cache_native PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)
set_target_properties(block_tree_cache PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 2
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES)

install(TARGETS block_tree_cache
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES include/block_tree_cache.h include/block_tree_cache.hpp include/block_tree_cache_worker.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if(BUILD_TESTING)
    add_executable(block_tree_cache_contract tests/contract/Contract.cc)
    target_link_libraries(block_tree_cache_contract PRIVATE block_tree_cache)
    add_test(NAME block_tree_cache_contract COMMAND block_tree_cache_contract)
    add_executable(block_tree_cache_remote_prefetch_pressure tests/contract/RemotePrefetchPressureContract.cc)
    target_link_libraries(block_tree_cache_remote_prefetch_pressure PRIVATE block_tree_cache)
    add_test(NAME block_tree_cache_remote_prefetch_pressure COMMAND block_tree_cache_remote_prefetch_pressure)
    add_executable(block_tree_cache_rank_worker_abi tests/contract/RankWorkerContract.cc)
    target_link_libraries(block_tree_cache_rank_worker_abi PRIVATE block_tree_cache Threads::Threads)
    add_test(NAME block_tree_cache_rank_worker_abi COMMAND block_tree_cache_rank_worker_abi)
    function(add_native_contract target source)
        add_executable(${target} tests/contract/${source} $<TARGET_OBJECTS:block_tree_cache_native>)
        target_compile_features(${target} PRIVATE cxx_std_17)
        target_include_directories(${target} PRIVATE . include "${BTC_REPO_ROOT}" "${CMAKE_CURRENT_SOURCE_DIR}/support")
        target_link_libraries(${target} PRIVATE Threads::Threads)
        add_test(NAME ${target} COMMAND ${target})
    endfunction()
    add_native_contract(block_tree_cache_native_tree_contract NativeTreeContract.cc)
    add_native_contract(block_tree_cache_multi_rank_contract MultiRankContract.cc)
    add_native_contract(block_tree_cache_native_worker_contract NativeWorkerContract.cc)
    find_package(Python3 3.10 QUIET COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        foreach(contract IN ITEMS process_receipts group_matching host_region tiered_eviction)
            add_test(NAME block_tree_cache_${contract}
                COMMAND "${CMAKE_COMMAND}" -E env "BLOCK_TREE_CACHE_LIBRARY=$<TARGET_FILE:block_tree_cache>"
                    "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/bindings/python" "PYTHONDONTWRITEBYTECODE=1"
                    "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/contract/test_${contract}.py")
        endforeach()
    endif()
endif()
