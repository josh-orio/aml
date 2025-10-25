BUILD_DIR := build
EXECUTABLE := $(BUILD_DIR)/statlib

.PHONY: all install

all: test

test:
	@clear
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake -D CMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON .. && $(MAKE) -j && make test ARGS="--output-on-failure"

install:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake -D CMAKE_BUILD_TYPE=Release .. && sudo $(MAKE) -j install

clean:
	@rm -rf $(BUILD_DIR) .cache
