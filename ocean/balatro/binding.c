#include "balatro.h"
#define OBS_SIZE 1611
#define NUM_ATNS 1
#define ACT_SIZES {DISCARD + 1}
#define OBS_TENSOR_T FloatTensor


#define Env Balatro
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
	env->num_agents = 1;
    env->reward_completed_blind = dict_get(kwargs, "reward_completed_blind")->value;
    env->reward_unused_hand = dict_get(kwargs, "reward_unused_hand")->value;
    env->reward_failure_penalty = dict_get(kwargs, "reward_failure_penalty")->value;
    env->reward_illegal_action_penalty = dict_get(kwargs, "reward_illegal_action_penalty")->value;
    env->reward_deselect_penalty = dict_get(kwargs, "reward_deselect_penalty")->value;
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->perf);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "win", log->win);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "filler", 123456789.0);
    dict_set(out, "remaining_hands", log->remaining_hands);
    dict_set(out, "remaining_discards", log->remaining_discards);
    dict_set(out, "num_played_cards", log->num_played_cards);
    dict_set(out, "num_discarded_cards", log->num_discarded_cards);
    dict_set(out, "played_hand_type_none", log->played_hand_type_none);
    dict_set(out, "discarded_hand_type_none", log->discarded_hand_type_none);
    dict_set(out, "played_hand_type_high_card", log->played_hand_type_high_card);
    dict_set(out, "discarded_hand_type_high_card", log->discarded_hand_type_high_card);
    dict_set(out, "played_hand_type_pair", log->played_hand_type_pair);
    dict_set(out, "discarded_hand_type_pair", log->discarded_hand_type_pair);
    dict_set(out, "played_hand_type_two_pair", log->played_hand_type_two_pair);
    dict_set(out, "discarded_hand_type_two_pair", log->discarded_hand_type_two_pair);
    dict_set(out, "played_hand_type_three_of_a_kind", log->played_hand_type_three_of_a_kind);
    dict_set(out, "discarded_hand_type_three_of_a_kind", log->discarded_hand_type_three_of_a_kind);
    dict_set(out, "played_hand_type_straight", log->played_hand_type_straight);
    dict_set(out, "discarded_hand_type_straight", log->discarded_hand_type_straight);
    dict_set(out, "played_hand_type_flush", log->played_hand_type_flush);
    dict_set(out, "discarded_hand_type_flush", log->discarded_hand_type_flush);
    dict_set(out, "played_hand_type_full_house", log->played_hand_type_full_house);
    dict_set(out, "discarded_hand_type_full_house", log->discarded_hand_type_full_house);
    dict_set(out, "played_hand_type_four_of_a_kind", log->played_hand_type_four_of_a_kind);
    dict_set(out, "discarded_hand_type_four_of_a_kind", log->discarded_hand_type_four_of_a_kind);
    dict_set(out, "played_hand_type_straight_flush", log->played_hand_type_straight_flush);
    dict_set(out, "discarded_hand_type_straight_flush", log->discarded_hand_type_straight_flush);
    dict_set(out, "played_hand_type_five_of_a_kind", log->played_hand_type_five_of_a_kind);
    dict_set(out, "discarded_hand_type_five_of_a_kind", log->discarded_hand_type_five_of_a_kind);
    dict_set(out, "played_hand_type_flush_house", log->played_hand_type_flush_house);
    dict_set(out, "discarded_hand_type_flush_house", log->discarded_hand_type_flush_house);
    dict_set(out, "played_hand_type_flush_five", log->played_hand_type_flush_five);
    dict_set(out, "discarded_hand_type_flush_five", log->discarded_hand_type_flush_five);
}
