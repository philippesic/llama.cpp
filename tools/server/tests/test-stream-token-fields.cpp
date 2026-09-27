#include "server-task.h"
#include <cassert>
#include <iostream>
int main() {
    server_task_result_cmpl_partial partial{};
    partial.n_decoded = 2;
    partial.tokens = {10, 11};
    partial.return_tokens = true;
    auto chunks = partial.to_json_oaicompat_chat();
    assert(chunks.size() == 1);
    assert(chunks[0]["tokens"] == json::array({10, 11}));
    assert(chunks[0]["choices"][0]["delta"].empty());
    partial.return_tokens = false;
    assert(partial.to_json_oaicompat_chat().empty());
    server_task_result_cmpl_final final{};
    final.tokens = {10, 11, 12};
    final.generation_params.return_tokens = true;
    auto ending = final.to_json_oaicompat_chat_stream();
    assert(ending.back()["generated_token_ids"] == json::array({10, 11, 12}));
    final.generation_params.return_tokens = false;
    assert(!final.to_json_oaicompat_chat_stream().back().contains("generated_token_ids"));
    std::cout << "4 native serializer gates passed\n";
}
