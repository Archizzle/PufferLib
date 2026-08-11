#include "yahtzee.h"
#define OBS_SIZE 37  // 6 dice distribution + 1 rerolls remaining + 13 empty boxes + 13 score per box + 1 top section capped + 1 yahtzee bonus state + 1 yahtzee bonus count + 1 turns remaining
#define NUM_ATNS (7) // 6 (1 for each die face) + 1 for boxes
#define ACT_SIZES {NUM_DICE+1, NUM_DICE+1, NUM_DICE+1, NUM_DICE+1, NUM_DICE+1, NUM_DICE+1, NUM_BOXES + 1}
#define OBS_TENSOR_T FloatTensor
#define MY_ACTION_MASK (NUM_DICE_ATNS + NUM_BOXES + 1)

#define Env Yahtzee
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
}
