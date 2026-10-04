#include "common.h"
#include "speculative.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

static void author_mode(bool enabled) {
#ifdef _WIN32
    _putenv_s("DSPARK_REQUIRE_AUTHOR_LAYOUT", enabled ? "1" : "");
#else
    if (enabled) setenv("DSPARK_REQUIRE_AUTHOR_LAYOUT", "1", 1);
    else unsetenv("DSPARK_REQUIRE_AUTHOR_LAYOUT");
#endif
}

static void check(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

static common_params draft_params(common_speculative_type type, int n, int sequences, bool backend) {
    common_params params;
    params.n_batch = params.n_ubatch = 32;
    params.n_parallel = sequences;
    params.speculative.types = {type};
    params.speculative.draft.mparams.path = "not-loaded.gguf";
    params.speculative.draft.n_max = n;
    params.speculative.draft.backend_sampling = backend;
    return common_base_params_to_speculative(params);
}

int main() {
    try {
        author_mode(false);
        auto normal = draft_params(COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK, 3, 1, true);
        check(normal.n_outputs_max == 4 && normal.n_outputs_max_per_seq == 4, "default reservation changed");
        author_mode(true);
        auto short_block = draft_params(COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK, 3, 1, true);
        check(short_block.n_outputs_max >= 7 && short_block.n_outputs_max_per_seq >= 7, "seven noise outputs do not fit short proposal");
        auto maximum = draft_params(COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK, 7, 1, true);
        check(maximum.n_outputs_max == 8 && maximum.n_outputs_max_per_seq == 8, "maximum reservation changed");
        auto multiple = draft_params(COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK, 3, 2, true);
        check(multiple.n_outputs_max == 14 && multiple.n_outputs_max_per_seq == 7, "multiple sequence reservation incomplete");
        auto cpu_sampling = draft_params(COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK, 3, 1, false);
        check(cpu_sampling.n_outputs_max == 7 && cpu_sampling.n_outputs_max_per_seq == 1, "raw outputs or CPU sampling policy changed");
        auto eagle = draft_params(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, 5, 1, true);
        check(eagle.n_outputs_max == 1 && eagle.n_outputs_max_per_seq == 1, "EAGLE reservation changed");
        auto target = common_speculative_get_output_limits(32, 1, 3);
        check(target.total == 4 && target.per_seq == 4, "target verifier reservation changed");
        author_mode(false);
        std::puts("DSpark reservation checks passed (7 cases)");
        return 0;
    } catch (const std::exception & error) {
        author_mode(false);
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
