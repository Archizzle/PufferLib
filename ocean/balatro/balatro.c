/* Pure C demo file for Squared. Build it with:
 * bash scripts/build_ocean.sh target local (debug)
 * bash scripts/build_ocean.sh target fast
 * We suggest building and debugging your env in pure C first. You
 * get faster builds and better error messages. To keep this example
 * simple, it does not include C neural nets. See Target for that.
 */

#include "balatro.h"

void demo() {
    // Balatro env = {.seed = 42};
    Balatro env = {.seed = 0};
    // uint16_t observation_size = NUM_UNDRAWN_OBS * MAX_DECK_SIZE + NUM_HAND_OBS * MAX_HAND_SIZE + NUM_DISCARD_OBS * MAX_DECK_SIZE + 6; // 6 for extra observations currently
    uint16_t observation_size = 1611; // 6 for extra observations currently

    env.observations = (float*)calloc(observation_size, sizeof(float));
    env.actions = (float*)calloc(1, sizeof(float));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (float*)calloc(1, sizeof(float));

    c_reset(&env);
    c_render(&env);
    while (!WindowShouldClose()) {
        if (! IsKeyDown(KEY_LEFT_SHIFT)) {
            env.actions[0] = 0;
            if (IsKeyDown(KEY_ONE) || IsKeyDown(KEY_Q)) env.actions[0] = 0 + SELECT_FIRST_CARD;
            if (IsKeyDown(KEY_TWO) || IsKeyDown(KEY_W)) env.actions[0] = 1 + SELECT_FIRST_CARD;
            if (IsKeyDown(KEY_THREE) || IsKeyDown(KEY_E)) env.actions[0] = 2 + SELECT_FIRST_CARD;
            if (IsKeyDown(KEY_FOUR) || IsKeyDown(KEY_R)) env.actions[0] = 3 + SELECT_FIRST_CARD;
            if (IsKeyDown(KEY_FIVE) || IsKeyDown(KEY_T)) env.actions[0] = 4 + SELECT_FIRST_CARD;
            if (IsKeyDown(KEY_SIX) || IsKeyDown(KEY_T)) env.actions[0] = 5 + SELECT_FIRST_CARD;
            if (IsKeyDown(KEY_SEVEN) || IsKeyDown(KEY_U)) env.actions[0] = 6 + SELECT_FIRST_CARD;
            if (IsKeyDown(KEY_EIGHT) || IsKeyDown(KEY_I)) env.actions[0] = 7 + SELECT_FIRST_CARD;
            if (IsKeyDown(KEY_ENTER)) env.actions[0] = PLAY_HAND; // Discard all and redraw
            if (IsKeyDown(KEY_SPACE)) env.actions[0] = DISCARD; // Discard all and redraw
            // if (IsKeyDown(KEY_UP)    || IsKeyDown(KEY_W)) env.actions[0] = UP;
            // if (IsKeyDown(KEY_DOWN)  || IsKeyDown(KEY_S)) env.actions[0] = DOWN;
            // if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)) env.actions[0] = LEFT;
            // if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.actions[0] = RIGHT;

        } else {
            env.actions[0] = 1 + rand() % DISCARD;
        }
        c_step(&env);
        c_render(&env);
    
    }
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
}

int main() {
	demo();
	return 0;
}
