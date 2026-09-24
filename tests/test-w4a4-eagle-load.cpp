#include "llama.h"

#include <cstdio>
#include <cstring>

int main(int argc, char ** argv) {
    if (argc != 3 || (strcmp(argv[1], "--accept") != 0 && strcmp(argv[1], "--reject") != 0)) {
        fprintf(stderr, "usage: %s --accept|--reject draft.gguf\n", argv[0]);
        return 2;
    }
    const bool expect_load = strcmp(argv[1], "--accept") == 0;
    llama_backend_init();
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[2], params);
    const bool loaded = model != nullptr;
    if (model) llama_model_free(model);
    llama_backend_free();
    if (loaded != expect_load) {
        fprintf(stderr, "W4A4 model load result disagreed with expectation\n");
        return 1;
    }
    return 0;
}
