CXX ?= c++
PYTHON ?= python3
UV ?= uv
DATA_DIR ?= bench/data
BASELINE ?= bench/official_baseline.json
export DATA_DIR CXX

BUILD := build
HEADERS := $(wildcard include/xhftbacktest/*.hpp)
COMMON := -std=c++20 -Wall -Wextra -Wpedantic -Werror -ffp-contract=off -pthread
LTO ?= -flto=auto
export LTO
RELEASE := -O3 $(LTO) -DNDEBUG
UBTRAP := -O3 $(LTO) -fsanitize=undefined -fsanitize-trap=undefined -D_GLIBCXX_ASSERTIONS
COMPILE = $(CXX) $(CPPFLAGS) $(CXXFLAGS) -Iinclude $(COMMON)
BENCH_SRC := bench/bench_parity_and_speed.cpp

.DELETE_ON_ERROR:
.PHONY: all test bench check python test-multi test-python clean FORCE
all: $(BUILD)/test_smoke

$(BUILD):
	mkdir -p $@

$(BUILD)/test_smoke: tests/test_smoke.cpp $(HEADERS) Makefile | $(BUILD)
	$(COMPILE) $(RELEASE) $< $(LDFLAGS) -o $@

$(BUILD)/test_smoke_ubsan: tests/test_smoke.cpp $(HEADERS) Makefile | $(BUILD)
	$(COMPILE) $(UBTRAP) $< $(LDFLAGS) -o $@

test: $(BUILD)/test_smoke $(BUILD)/test_smoke_ubsan
	./$(BUILD)/test_smoke
	./$(BUILD)/test_smoke_ubsan

# Generate constants from the chosen dataset and oracle without a C++ JSON library.
define READ_BASELINE
import json, sys
from pathlib import Path
baseline, meta = (json.loads(Path(p).read_text()) for p in sys.argv[1:])
if baseline["hftbacktest_version"] != "2.4.4" or baseline["n_events"] != meta["n_events"]:
    sys.exit("Expected a version 2.4.4 baseline for this dataset")
tick = meta["tick_size"]
lo, hi = (round(meta[k] / tick) for k in ("roi_lb", "roi_ub"))
print(f'constexpr std::size_t event_count = {baseline["n_events"]};')
print(f'constexpr double tick = {tick};')
print(f'constexpr AssetConfig asset{{tick, {meta["lot_size"]}, {lo}, {hi}, 1., -0.00003, 0.0005}};')
fields = ("position", "balance", "fee", "num_trades", "trading_volume", "equity_at_last_mid", "final_mid")
print("constexpr Reference refs[] = {")
for model in ("risk_adverse", "power_prob"):
    sample = baseline["models"][model]
    values = [sample["final_state"][k] for k in fields] + [sample["median"]["events_per_sec"]]
    print("    {" + ", ".join(map(str, values)) + "},")
print("};")
endef
export READ_BASELINE

# DATA_DIR and BASELINE can switch between invocations, even to older files.
$(BUILD)/baseline.inc: FORCE | $(BUILD)
	@"$(PYTHON)" -c "$$READ_BASELINE" "$(BASELINE)" "$(DATA_DIR)/lista_real_meta.json" > $@.tmp
	@if cmp -s $@.tmp $@; then rm $@.tmp; else mv $@.tmp $@; fi

$(BUILD)/bench_release: $(BENCH_SRC) $(HEADERS) $(BUILD)/baseline.inc Makefile
	$(COMPILE) $(RELEASE) -march=native -I$(BUILD) $< $(LDFLAGS) -o $@

$(BUILD)/bench_ubsan: $(BENCH_SRC) $(HEADERS) $(BUILD)/baseline.inc Makefile
	$(COMPILE) $(UBTRAP) -march=native -I$(BUILD) $< $(LDFLAGS) -o $@

bench: $(BUILD)/bench_release
	./$< $(BUILD)/cpp_results.json

check: $(BUILD)/bench_ubsan
	./$< $(BUILD)/ubsan_results.json

# Query Python only for targets that use the extension or the Rust reference.
ifneq ($(filter python test-python test-multi,$(MAKECMDGOALS)),)
PY_RUN := "$(PYTHON)"
PY_CHECK := import numpy; assert int(numpy.__version__.split(".")[0]) >= 2
UV_DEPS := --with 'numpy>=2'
ifneq ($(filter python test-python,$(MAKECMDGOALS)),)
PY_CHECK += ; import nanobind
UV_DEPS += --with nanobind
endif
ifneq ($(filter test-python test-multi,$(MAKECMDGOALS)),)
PY_CHECK += ; import hftbacktest, numba; assert hftbacktest.__version__ == "2.4.4"
UV_DEPS += --with hftbacktest==2.4.4
endif
ifeq ($(shell "$(PYTHON)" -B -c '$(PY_CHECK)' >/dev/null 2>&1 && echo ok),)
ifeq ($(shell command -v "$(UV)" 2>/dev/null),)
$(error Missing Python dependencies. Install the packages listed in README.md or set UV to the uv executable)
endif
PY_RUN := "$(UV)" run --no-project --python 3.12 $(UV_DEPS) python
endif
endif

ifneq ($(filter python test-python,$(MAKECMDGOALS)),)
EXT := $(shell $(PY_RUN) -B -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
ifneq ($(.SHELLSTATUS),0)
$(error Could not query the Python extension build environment)
endif
MODULE := $(BUILD)/xhftbacktest$(EXT)

# Discover headers in the selected interpreter. Pass paths as individual arguments
# so virtual environments can live anywhere, including directories with spaces.
define BUILD_MODULE
import os, sys, sysconfig
from pathlib import Path
import nanobind as nb
import numpy as np
src = Path(nb.source_dir())
flags = []
for path, label in ((sysconfig.get_path("include"), "python/include"),
                    (np.get_include(), "numpy/include"), (nb.include_dir(), "nanobind/include")):
    flags.extend(["-isystem", path, f"-ffile-prefix-map={path}={label}"])
flags.extend(["-I", str(src.parent / "ext/robin_map/include"), f"-ffile-prefix-map={src.parent}=nanobind"])
command = [*sys.argv[1:], *flags, str(src / "nb_combined.cpp")]
os.execvp(command[0], command)
endef

$(MODULE): export BUILD_MODULE := $(BUILD_MODULE)
$(MODULE): python/xhftbacktest_py.cpp $(HEADERS) Makefile | $(BUILD)
	$(PY_RUN) -B -c "$$BUILD_MODULE" $(COMPILE) $(RELEASE) -fPIC -fvisibility=hidden -fno-strict-aliasing -shared $< $(LDFLAGS) -o $@

python: $(MODULE)
endif

test-python: python
	$(PY_RUN) -B python/test_python_parity.py

test-multi:
	$(PY_RUN) -B bench/test_multi_asset_parity.py

clean:
	rm -rf $(BUILD)
