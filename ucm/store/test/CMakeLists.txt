if(BUILD_UNIT_TESTS)
    include(GoogleTest)
    file(GLOB_RECURSE UCMSTORE_TEST_SOURCE_FILES "./case/*.cc")
    add_executable(ucmstore.test ${UCMSTORE_TEST_SOURCE_FILES})
    target_include_directories(ucmstore.test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/case
        ${CMAKE_BINARY_DIR}/_deps/googletest-src/googletest/include
        ${CMAKE_BINARY_DIR}/_deps/googletest-src/googlemock/include
    )
    target_link_directories(ucmstore.test PRIVATE ${CMAKE_BINARY_DIR}/lib)
    target_link_libraries(ucmstore.test PRIVATE
        posixstore cachestore fakestore pcstore compressor
        gtest_main gtest gmock
    )
    gtest_discover_tests(ucmstore.test)
endif()
