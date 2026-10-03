CXX ?= clang++
CXXFLAGS ?= -std=c++17 -O3 -DNDEBUG -Wall -Wextra -Wpedantic
LDFLAGS ?=
LDLIBS ?= -lz
SOXR_CFLAGS ?= $(shell pkg-config --cflags soxr)
SOXR_LIBS ?= $(shell pkg-config --libs soxr)
CXXFLAGS += $(SOXR_CFLAGS)
LDLIBS += $(SOXR_LIBS)
SDKROOT ?= $(shell xcrun --show-sdk-path 2>/dev/null)
BUILD_DIR ?= build
TARGET ?= $(BUILD_DIR)/litevox
DIST_DIR ?= dist
DIST_TARGET ?= $(DIST_DIR)/litevox
BUNDLE_DIR ?= $(DIST_DIR)/bootstrap-bundle
BUNDLE_ARCHIVE ?= $(DIST_DIR)/bootstrap-bundle.tar.gz
BUNDLE_ARCHIVE_SHA256 ?= $(BUNDLE_ARCHIVE).sha256
OPEN_JTALK_DIR ?= $(BUILD_DIR)/openjtalk
OPEN_JTALK_INCLUDE_DIR ?= $(OPEN_JTALK_DIR)/include
OPEN_JTALK_LIB ?= $(OPEN_JTALK_DIR)/lib/libopenjtalk.a
SRC := $(sort $(wildcard src/*.cpp))
OBJ := $(patsubst src/%.cpp,$(BUILD_DIR)/%.o,$(SRC))
DEP := $(OBJ:.o=.d)
MISSING_DEP := $(filter-out $(wildcard $(DEP)),$(DEP))
UNAME_S := $(shell uname -s)

ifneq ($(SDKROOT),)
CXXFLAGS += -isysroot $(SDKROOT) -isystem $(SDKROOT)/usr/include/c++/v1
endif

CXXFLAGS += -I$(OPEN_JTALK_INCLUDE_DIR)
LDLIBS += $(OPEN_JTALK_LIB)

all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET): $(OBJ) $(OPEN_JTALK_LIB)
	$(CXX) $(OBJ) -o $@ $(LDFLAGS) $(LDLIBS)

$(BUILD_DIR)/openjtalk-src/src/CMakeLists.txt: | $(BUILD_DIR)
	git clone --depth 1 --branch 1.11 https://github.com/VOICEVOX/open_jtalk.git $(BUILD_DIR)/openjtalk-src

ifeq ($(OPEN_JTALK_LIB),$(OPEN_JTALK_DIR)/lib/libopenjtalk.a)
$(OPEN_JTALK_LIB): $(BUILD_DIR)/openjtalk-src/src/CMakeLists.txt
	cmake -S $(BUILD_DIR)/openjtalk-src/src -B $(BUILD_DIR)/openjtalk-cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=$(abspath $(OPEN_JTALK_DIR))
	+cmake --build $(BUILD_DIR)/openjtalk-cmake --parallel
	cmake --install $(BUILD_DIR)/openjtalk-cmake
endif

$(BUILD_DIR)/native_text_query.o: $(OPEN_JTALK_LIB)

$(BUILD_DIR)/%.o: src/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/Compatibility: tests/Compatibility.cpp $(filter-out $(BUILD_DIR)/main.o,$(OBJ)) $(OPEN_JTALK_LIB)
	$(CXX) $(CXXFLAGS) -UNDEBUG -Isrc $< $(filter-out $(BUILD_DIR)/main.o,$(OBJ)) -o $@ $(LDFLAGS) $(LDLIBS)

check: $(BUILD_DIR)/Compatibility
	./$(BUILD_DIR)/Compatibility $(if $(RUNTIME_ROOT),"$(RUNTIME_ROOT)",)

dist: $(TARGET)
	mkdir -p $(DIST_DIR)
	cp $(TARGET) $(DIST_TARGET)
	if [ -d resources ]; then mkdir -p $(DIST_DIR)/resources; cp resources/* $(DIST_DIR)/resources/; fi
ifeq ($(UNAME_S),Darwin)
	codesign --force --sign - $(DIST_TARGET)
endif

clean:
	rm -rf $(BUILD_DIR)

verify-runtime-from-archives:
	test -n "$(VOICEVOX_ZIP)"
	test -n "$(ONNXRUNTIME_ARCHIVE)"
	./tools/verify-runtime-from-archives.sh "$(VOICEVOX_ZIP)" "$(ONNXRUNTIME_ARCHIVE)" "$(if $(RUNTIME_ROOT),$(RUNTIME_ROOT),/tmp/litevox-runtime-from-archives-verify)" "$(if $(RESULT_PREFIX),$(RESULT_PREFIX),audio-compare/runtime-extract-one-step-verify)"

verify-cli-smoke: dist
	test -n "$(RUNTIME_ROOT)" || export RUNTIME_ROOT="$(DIST_DIR)"; \
	test -n "$(RESULT_PREFIX)" || export RESULT_PREFIX="audio-compare/cli-smoke"; \
	sh ./tools/verify-cli-smoke.sh "$${RUNTIME_ROOT}" "$${RESULT_PREFIX}"

verify-bootstrap-bundle: bootstrap-bundle-archive
	cd $(BUNDLE_DIR) && shasum -a 256 -c SHA256SUMS
	cd $(DIST_DIR) && shasum -a 256 -c $(notdir $(BUNDLE_ARCHIVE_SHA256))

bootstrap-bundle: dist
	rm -rf $(BUNDLE_DIR)
	mkdir -p $(BUNDLE_DIR)/tools $(BUNDLE_DIR)/resources
	cp $(DIST_TARGET) $(BUNDLE_DIR)/litevox
	cp README.bundle.md $(BUNDLE_DIR)/README.md
	cp tools/write_bundle_manifest.py $(BUNDLE_DIR)/tools/
	cp tools/verify-cli-smoke.sh $(BUNDLE_DIR)/tools/
	cp tools/verify-runtime-from-archives.sh $(BUNDLE_DIR)/tools/
	cp tools/compare_voicevox_http.py $(BUNDLE_DIR)/tools/
	cp tools/compare_voicevox_http_full.py $(BUNDLE_DIR)/tools/
	cp $(DIST_DIR)/resources/openapi.json $(BUNDLE_DIR)/resources/openapi.json
	chmod +x $(BUNDLE_DIR)/tools/verify-runtime-from-archives.sh $(BUNDLE_DIR)/tools/verify-cli-smoke.sh $(BUNDLE_DIR)/tools/write_bundle_manifest.py
ifeq ($(UNAME_S),Darwin)
	codesign --force --sign - $(BUNDLE_DIR)/litevox
endif
	python3 ./tools/write_bundle_manifest.py $(BUNDLE_DIR)

bootstrap-bundle-archive: bootstrap-bundle
	rm -f $(BUNDLE_ARCHIVE)
	tar -C $(dir $(BUNDLE_DIR)) -czf $(BUNDLE_ARCHIVE) $(notdir $(BUNDLE_DIR))
	shasum -a 256 $(BUNDLE_ARCHIVE) | awk '{print $$1 "  $(notdir $(BUNDLE_ARCHIVE))"}' > $(BUNDLE_ARCHIVE_SHA256)

.PHONY: all check clean dist verify-runtime-from-archives verify-cli-smoke verify-bootstrap-bundle bootstrap-bundle bootstrap-bundle-archive force-dep-rebuild

-include $(DEP)

ifneq ($(MISSING_DEP),)
$(OBJ): force-dep-rebuild

force-dep-rebuild:
endif
