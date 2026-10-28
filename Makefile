# Native tools. Use an ARM hard-float glibc compiler for the tested Debian RV1106.
CC ?= cc
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Werror
BUILD_DIR ?= build
RKNN_INCLUDE ?= third_party/rknn/include
RKNN_LIB ?= third_party/rknn/lib
NATIVE := speaker_verification/native

.PHONY: all cpu test models-check
all: cpu $(BUILD_DIR)/rv1106_bench
cpu: $(BUILD_DIR)/fbank $(BUILD_DIR)/eres2net_tail

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/fbank: $(NATIVE)/fbank.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LDFLAGS) -lm -o $@

$(BUILD_DIR)/eres2net_tail: $(NATIVE)/eres2net_tail.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LDFLAGS) -lm -o $@

$(BUILD_DIR)/rv1106_bench: $(NATIVE)/rv1106_bench.c | $(BUILD_DIR)
	@test -f "$(RKNN_INCLUDE)/rknn_api.h" || { echo 'Provide RKNN_INCLUDE with matching rknn_api.h'; exit 1; }
	@test -f "$(RKNN_LIB)/librknnmrt.so" || { echo 'Provide RKNN_LIB with board-compatible librknnmrt.so'; exit 1; }
	$(CC) $(CPPFLAGS) $(CFLAGS) -I"$(RKNN_INCLUDE)" $< $(LDFLAGS) -L"$(RKNN_LIB)" -lrknnmrt -lm -Wl,-rpath,'$$ORIGIN' -o $@

models-check:
	python3 -m speaker_verification.verify_models

test:
	python3 -m unittest discover -s tests -p 'test_*.py' -v
	node tests/test_web_ui.js
