#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "yahtzee.h"
// #include "puffernet.h"

void demo() {
    // TODO: Fill this in later, e.g.
    // Weights* weights = load_weights("resources/yahtzee/yahtzee_weights.bin");

    float observations[37] = {0};
    float actions[7] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    unsigned char action_mask[NUM_DICE_ATNS + NUM_BOXES + 1] = {0};

    Yahtzee env = {
        .num_agents = 1,
        .observations = observations,
        .actions = actions,
        .rewards = rewards,
        .terminals = terminals,
        .action_mask = action_mask,
        .rng = (unsigned int)time(NULL),
    };

    c_reset(&env);
    for (int ep = 0; ep < 10;) {
        int offset = 0;
        int sizes[7] = {6, 6, 6, 6, 6, 6, NUM_BOXES + 1};
        for (int h = 0; h < 7; h++) {
            do {
                env.actions[h] = (float)(rand_r(&env.rng) % sizes[h]);
            } while (!env.action_mask[offset + (int)env.actions[h]]);
            offset += sizes[h];
        }

        c_step(&env);
        if (env.terminals[0]) {
            printf("episode %d score=%d\n", ++ep, env.total_score);
            c_reset(&env);
        }
    }
}

int main() {
    demo();
}
