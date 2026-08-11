/* Yahtzee: a sample multiagent env about puffers eating stars.
 * Use this as a tutorial and template for your own multiagent envs.
 * We suggest starting with the Squared env for a simpler intro.
 * Star PufferLib on GitHub to support. It really, really helps!
 */

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "raylib.h"


#define NUM_DICE 5
#define NUM_TURNS 13
#define NUM_ROLLS 3

#define NUM_DICE_ATNS ((NUM_DICE + 1) * 6)

#define FIXED_FULL_HOUSE_SCORE 25
#define FIXED_SMALL_STRAIGHT_SCORE 30
#define FIXED_LARGE_STRAIGHT_SCORE 40
#define FIXED_YAHTZEE_SCORE 50

#define BONUS_TOP_THRESHOLD 63
#define BONUS_TOP_SCORE 35
#define BONUS_YAHTZEE_SCORE 100

enum Boxes {
    ONES,
    TWOS,
    THREES,
    FOURS,
    FIVES,
    SIXES,
    THREE_OF_A_KIND,
    FOUR_OF_A_KIND,
    FULL_HOUSE,
    SMALL_STRAIGHT,
    LARGE_STRAIGHT,
    YAHTZEE,
    CHANCE,
    NUM_BOXES,
};

static const char roll_no_dice[NUM_DICE_ATNS] = {
    1, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0
};
// Required struct. Only use floats!
typedef struct {
    float perf; // Recommended 0-1 normalized single real number perf metric
    float score; // Recommended unnormalized single real number perf metric
    float episode_return; // Recommended metric: sum of agent rewards over episode
    float episode_length; // Recommended metric: number of steps of agent episode
    // Any extra fields you add here may be exported to Python in binding.c
    float n; // Required as the last field
} Log;

typedef struct {
    float x;
    float y;
    float heading;
    float speed;
    int ticks_since_reward;
} Agent;

// Required that you have some struct for your env
// Recommended that you name it the same as the env file
typedef struct {
    Log log; // Required field. Env binding code uses this to aggregate logs
    float* observations; // Required. You can use any obs type, but make sure it matches in Python!
    float* actions; // Required
    float* rewards; // Required
    float* terminals; // Required
    int num_agents; // Required
    unsigned char* action_mask;

    // ===== Dice State =====
    // DECISION: 3 options: 
    //      Individual dice, unsorted
    //      Individual dice, sorted
    //      Buckets of frequency (Current)
    int dice_value_distribution[6];
    int num_rerolls_remaining;          // 2 rerolls = already rolled once, 
                                        // 1 reroll = already rolled twice, 
                                        // 0 rerolls = musch choose box to place in

    // ===== Boxes State ======
    char empty_boxes[NUM_BOXES];        // 0 if filled, 1 if empty
                                        // DECISION: either use bitmask, can replace with int/bool array or get rid of entirely due to socre_per_box;
    int score_per_box[NUM_BOXES];       // DECISION: -1 for empty boxes, or use filled boxes_mask and let 0 equal either unfilled or filled with 0 score
    int top_section_capped;             // min(top_section_score, 63); if 63, we get bonus
    int yahtzee_bonus_state;                  // 0 if yahtzee box empty, 1 if filled with scored with yahtzee, -1 if zeroed
                                        // Computed by empty_boxes[YAHTZEE] != 0 && score_per_box[YAHTZEE] > 0
                                        // DECISION: how to signify eligible vs not eligible? use score_per_box or other helper variable?
    int yahtzee_bonus_count;
    int total_score;
    int turns_remaining;

    // int width;
    // int height;
    unsigned int rng;
} Yahtzee;

// DECISION: Client struct to store history for display?
// Also, potentially need to keep track of which number coresponds to which index for freezing/gui

static inline void roll_all_dice(Yahtzee* env) {
    unsigned int turn_seed = rand_r(&env->rng);
    memset(env->dice_value_distribution, 0, sizeof(env->dice_value_distribution));
    for (int i = 0; i < NUM_DICE; i++) {
        int new_dice_value = rand_r(&turn_seed) % 6;
        env->dice_value_distribution[new_dice_value]++;
    }
}

/* Recommended to have an init function of some kind if you allocate
 * extra memory. This should be freed by c_close. Don't forget to call
 * this in binding.c!
 */
void init(Yahtzee* env) {
}

/* Recommended to have an observation function of some kind because
 * you need to compute agent observations in both reset and in step.
 * If using float obs, try to normalize to roughly -1 to 1 by dividing
 * by an appropriate constant.
 */
void compute_observations(Yahtzee* env) {
    int obs_idx = 0;
    
    for (int i = 0; i < 6; i++) {
        env->observations[obs_idx++] = (float)env->dice_value_distribution[i] / 6;
    }
    env->observations[obs_idx++] = (float)env->num_rerolls_remaining / (NUM_ROLLS - 1);
    
    for (int i = 0; i < NUM_BOXES; i++) {
        env->observations[obs_idx++] = (float)env->empty_boxes[i];
    }
    // DECISION: Technicallynot needed for future decisions
    for (int i = 0; i < NUM_BOXES; i++) {
        env->observations[obs_idx++] = (float)env->score_per_box[i] / FIXED_YAHTZEE_SCORE;
    }
    // DECISION: Include max score of unfilled boxes? Normalized by 50?

    env->observations[obs_idx++] = (float)env->top_section_capped / BONUS_TOP_THRESHOLD;
    env->observations[obs_idx++] = (float)env->yahtzee_bonus_state;
    env->observations[obs_idx++] = (float)env->yahtzee_bonus_count / NUM_TURNS;
    env->observations[obs_idx++] = (float)env->turns_remaining / NUM_TURNS;
   
    memset(env->action_mask, 0, NUM_DICE_ATNS + NUM_BOXES + 1);
    if (env->num_rerolls_remaining != 0) {
        int mask_idx = 0;
        for (int i = 0; i < 6; i++) {
            memset(&env->action_mask[mask_idx], 1, env->dice_value_distribution[i] + 1);
            mask_idx += (NUM_DICE + 1);
        }
        env->action_mask[NUM_DICE_ATNS + NUM_BOXES] = 1;
    } else {
        memcpy(env->action_mask, roll_no_dice, sizeof(roll_no_dice));
        memcpy(&env->action_mask[NUM_DICE_ATNS], env->empty_boxes, NUM_BOXES);
        
        bool rolled_yahtzee = false;
        for (int i = 0; i < 6; i++) {
            if (env->dice_value_distribution[i] == NUM_DICE) {
                rolled_yahtzee = true;
                break;
            }
        }
        // Bonus Yahtzee: need to have rolled a yahtzee, already filled the yahtzee square, and the score > 0
        if (rolled_yahtzee && env->empty_boxes[YAHTZEE] == 0 && env->score_per_box[YAHTZEE] > 0) {
            env->action_mask[NUM_DICE_ATNS + YAHTZEE] = 1;
        }
    }
}

// Required function
void c_reset(Yahtzee* env) {
    env->terminals[0] = 0;
    env->rewards[0] = 0.0;
    memset(env->dice_value_distribution, 0, sizeof(env->dice_value_distribution));
    env->dice_value_distribution[0] = NUM_DICE;

    roll_all_dice(env);

    env->num_rerolls_remaining = 2;
    memset(env->empty_boxes, 1, sizeof(env->empty_boxes));
    memset(env->score_per_box, 0, sizeof(env->score_per_box));
    env->top_section_capped = 0;
    env->yahtzee_bonus_state = 0;
    env->yahtzee_bonus_count = 0;
    env->total_score = 0;

    env->turns_remaining = NUM_TURNS;

    compute_observations(env);
}

// Required function
void c_step(Yahtzee* env) {
    env->rewards[0] = 0.0;
    if (env->num_rerolls_remaining != 0) {
        unsigned int turn_seed = rand_r(&env->rng);
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < (int)env->actions[i]; j++) {
                int new_dice_value = rand_r(&turn_seed) % 6;
                env->dice_value_distribution[i]--;
                env->dice_value_distribution[new_dice_value]++;
                // TODO: If client rendering stuff, ensure specific dice are rerolled
            }
        }
        env->num_rerolls_remaining--;
    } else {
        enum Boxes fill_box_idx = env->actions[6]; // 0-5 is for num of each dice face to reroll, idx 6 is which box to fill
        int new_score = 0;
        switch (fill_box_idx) {
            case ONES:
                new_score = env->dice_value_distribution[0] * 1;
                break;
            case TWOS:
                new_score = env->dice_value_distribution[1] * 2;
                break;
            case THREES:
                new_score = env->dice_value_distribution[2] * 3;
                break;
            case FOURS:
                new_score = env->dice_value_distribution[3] * 4;
                break;
            case FIVES:
                new_score = env->dice_value_distribution[4] * 5;
                break;
            case SIXES:
                new_score = env->dice_value_distribution[5] * 6;
                break;
            case THREE_OF_A_KIND: {
                for (int i = 0; i < 6; i++) {
                    if (env->dice_value_distribution[i] >= 3) {
                        for (int i = 0; i < 6; i++) {
                            new_score += env->dice_value_distribution[i] * (i + 1);
                        }
                        break;
                    }
                }
                break;
            }
            case FOUR_OF_A_KIND: {
                for (int i = 0; i < 6; i++) {
                    if (env->dice_value_distribution[i] >= 4) {
                        for (int i = 0; i < 6; i++) {
                            new_score += env->dice_value_distribution[i] * (i + 1);
                        }
                        break;
                    }
                }
                break;
            }
            case FULL_HOUSE: {
                int three_of_a_kind_face = -1;
                for (int i = 0; i < 6; i++) {
                    if (env->dice_value_distribution[i] == 3) {
                        three_of_a_kind_face = i;
                        break;
                    }
                }
                for (int i = 0; i < 6; i++) {
                    if (env->dice_value_distribution[i] == 2 && i != three_of_a_kind_face) {
                        new_score = FIXED_FULL_HOUSE_SCORE;
                        break;
                    }
                }
                break;
            }
            case SMALL_STRAIGHT:
                if (
                        (
                             env->dice_value_distribution[0] >= 1 &&
                             env->dice_value_distribution[1] >= 1 &&
                             env->dice_value_distribution[2] >= 1 &&
                             env->dice_value_distribution[3] >= 1
                        )
                        ||
                        (
                            env->dice_value_distribution[1] >= 1 &&
                            env->dice_value_distribution[2] >= 1 &&
                            env->dice_value_distribution[3] >= 1 &&
                            env->dice_value_distribution[4] >= 1
                        )
                        ||
                        (
                            env->dice_value_distribution[2] >= 1 &&
                            env->dice_value_distribution[3] >= 1 &&
                            env->dice_value_distribution[4] >= 1 &&
                            env->dice_value_distribution[5] >= 1
                        )
                    )
                        new_score = FIXED_SMALL_STRAIGHT_SCORE;
                break;

            case LARGE_STRAIGHT:
                if (
                    (
                        env->dice_value_distribution[0] >= 1 &&
                        env->dice_value_distribution[1] >= 1 &&
                        env->dice_value_distribution[2] >= 1 &&
                        env->dice_value_distribution[3] >= 1 &&
                        env->dice_value_distribution[4] >= 1
                    ) ||
                    (
                        env->dice_value_distribution[1] >= 1 &&
                        env->dice_value_distribution[2] >= 1 &&
                        env->dice_value_distribution[3] >= 1 &&
                        env->dice_value_distribution[4] >= 1 &&
                        env->dice_value_distribution[5] >= 1
                    )
                )
                    new_score = FIXED_LARGE_STRAIGHT_SCORE;
                break;
            case YAHTZEE: {
                bool is_yahtzee = false;
                for (int i = 0; i < 6; i++) {
                    if (env->dice_value_distribution[i] == NUM_DICE) {
                        is_yahtzee = true;
                        break;
                    }
                }
                if (!is_yahtzee) {
                    env->yahtzee_bonus_state = -1;
                    new_score = 0;
                } else {
                    if (env->yahtzee_bonus_state == 0) {
                        env->yahtzee_bonus_state = 1;
                        new_score = FIXED_YAHTZEE_SCORE;
                    } else {
                        env->yahtzee_bonus_count++;
                        new_score = BONUS_YAHTZEE_SCORE;
                    }
                }
                break;
            }
            case CHANCE:
                for (int i = 0; i < 6; i++) {
                    new_score += env->dice_value_distribution[i] * (i + 1);
                }
                break;
            default:
                break;
        }
        
        int old_score = env->total_score;
        env->score_per_box[fill_box_idx] = new_score;
        env->total_score += new_score;
        int old_top_total = env->top_section_capped;
        if (fill_box_idx <= SIXES) {
            env->top_section_capped = (BONUS_TOP_THRESHOLD < env->top_section_capped + new_score) ? BONUS_TOP_THRESHOLD : (env->top_section_capped + new_score);
        }
        if (old_top_total < BONUS_TOP_THRESHOLD && env->top_section_capped >= BONUS_TOP_THRESHOLD) {
            env->total_score += BONUS_TOP_SCORE;
        }

        env->rewards[0] = (float)(env->total_score - old_score) / FIXED_YAHTZEE_SCORE;
        env->log.episode_return += env->rewards[0];

        env->empty_boxes[fill_box_idx] = 0;
        roll_all_dice(env);
        env->num_rerolls_remaining = NUM_ROLLS - 1;
        env->turns_remaining--;
    }
    compute_observations(env);

    env->log.n++;

    if (env->turns_remaining == 0) {
        env->terminals[0] = 1;
        env->log.score = env->total_score;
        env->log.perf = env->total_score / 250.0;
    }
}

// Required function. Should handle creating the client on first call
void c_render(Yahtzee* env) {
    // if (env->client == NULL) {
    //     InitWindow(1080, 720, "PufferLib Yahtzee");
    //     SetTargetFPS(60);
    //     env->client = (Client*)calloc(1, sizeof(Client));
    //     env->client->puffer = LoadTexture("resources/shared/puffers_128.png");
    //     env->client->star = LoadTexture("resources/shared/star.png");
    // }
    //
    // // Standard across our envs so exiting is always the same
    // if (IsKeyDown(KEY_ESCAPE)) {
    //     exit(0);
    // }
    //
    // BeginDrawing();
    // ClearBackground((Color){6, 24, 24, 255});
    //
    // for (int i = 0; i < env->num_goals; i++) {
    //     Goal* goal = &env->goals[i];
    //     DrawTexture(env->client->star, (int)goal->x, (int)goal->y, WHITE);
    // }
    //
    // for (int i = 0; i < env->num_agents; i++) {
    //     Agent* agent = &env->agents[i];
    //     float heading = agent->heading;
    //     DrawTexturePro(
    //         env->client->puffer,
    //         (Rectangle){
    //             (heading < PI/2 || heading > 3*PI/2) ? 0 : 128,
    //             0, 128, 128,
    //         },
    //         (Rectangle){
    //             agent->x,
    //             agent->y,
    //             128,
    //             128
    //         },
    //         (Vector2){0, 0},
    //         0,
    //         WHITE
    //     );
    // }
    //
    // EndDrawing();
}

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(Yahtzee* env) {
    // free(env->agents);
    // free(env->goals);
    // if (env->client != NULL) {
    //     Client* client = env->client;
    //     UnloadTexture(client->puffer);
    //     UnloadTexture(client->star);
    //     CloseWindow();
    //     free(client);
    // }
}
