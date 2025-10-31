BUILD_DIR := build

.PHONY: all install

all: test

test:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake -D CMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBUILD_BENCHES=OFF .. && $(MAKE) -j && make test ARGS="--output-on-failure"

bench:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake -D CMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBUILD_BENCHES=ON .. && cmake --build . --config Release && ./benches/bench_example

run:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake -D CMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBUILD_BENCHES=OFF .. && $(MAKE) -j
	./build/example_program

install:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake -D CMAKE_BUILD_TYPE=Release .. && sudo $(MAKE) -j install

clean:
	@rm -rf $(BUILD_DIR) .cache
