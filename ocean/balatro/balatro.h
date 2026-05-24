/* Squared: a sample single-agent grid env.
 * Use this as a tutorial and template for your first env.
 * See the Target env for a slightly more complex example.
 * Star PufferLib on GitHub to support. It really, really helps!
 */

#include "raylib.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <math.h> // Used only for debugging/checking obs are not wrong.

const unsigned char NOOP = 0;
const unsigned char SELECT_FIRST_CARD = 1;
const unsigned char PLAY_HAND = 16;
const unsigned char DISCARD = 17;

const float TEMP_TARGET_SCORE = 300;
// const float TEMP_TARGET_SCORE = 600;

const uint8_t MAX_DECK_SIZE = 255;

const uint8_t MAX_HAND_SIZE = 15;
const uint8_t MAX_SELECTION_SIZE = 5;
const uint8_t STARTING_HAND_SIZE = 8;
// ---------------------------
// const uint8_t MAX_HAND_SIZE 5
// const uint8_t MAX_SELcECTION_SIZE 1    // Setting to 1 for now, but will
// change to 5 once we finish the basic env. const uint8_t STARTING_HAND_SIZE 5
// const uint8_t STARTING_NUM_DISCARDS = 15; // Initially can only discard card
// 1 per discard, will increase this in the future with cards that let you
// discard more
// ---------------------------

// White Stake
// const uint8_t STARTING_NUM_HANDS = 1;
// const uint8_t STARTING_NUM_DISCARDS = 10;

// Reshaped Gold Stake
const uint8_t STARTING_NUM_HANDS = 3;
const uint8_t STARTING_NUM_DISCARDS = 2;

const uint8_t NUM_UNDRAWN_OBS = 3;
const uint8_t NUM_HAND_OBS = 5;
const uint8_t NUM_DISCARD_OBS = 3;

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

// Required struct. Only use floats!
typedef struct {
  float perf;  // Recommended 0-1 normalized single real number perf metric
  float score; // Recommended unnormalized single real number perf metric
  float episode_return; // Recommended metric: sum of agent rewards over episode
  float episode_length; // Recommended metric: number of steps of agent episode
  // Any extra fields you add here may be exported to Python in binding.c
  float remaining_hands;
  float remaining_discards;
  float win;
  float num_played_cards;
  float num_discarded_cards;
  float played_hand_type_none;
  float played_hand_type_high_card;
  float played_hand_type_pair;
  float played_hand_type_two_pair;
  float played_hand_type_three_of_a_kind;
  float played_hand_type_straight;
  float played_hand_type_flush;
  float played_hand_type_full_house;
  float played_hand_type_four_of_a_kind;
  float played_hand_type_straight_flush;
  float played_hand_type_five_of_a_kind;
  float played_hand_type_flush_house;
  float played_hand_type_flush_five;
  float discarded_hand_type_none;
  float discarded_hand_type_high_card;
  float discarded_hand_type_pair;
  float discarded_hand_type_two_pair;
  float discarded_hand_type_three_of_a_kind;
  float discarded_hand_type_straight;
  float discarded_hand_type_flush;
  float discarded_hand_type_full_house;
  float discarded_hand_type_four_of_a_kind;
  float discarded_hand_type_straight_flush;
  float discarded_hand_type_five_of_a_kind;
  float discarded_hand_type_flush_house;
  float discarded_hand_type_flush_five;
  float n; // Required as the last field
} Log;

typedef enum {
  RANK_2 = 0,
  RANK_3,
  RANK_4,
  RANK_5,
  RANK_6,
  RANK_7,
  RANK_8,
  RANK_9,
  RANK_10,
  RANK_JACK,
  RANK_QUEEN,
  RANK_KING,
  RANK_ACE,
  RANK_COUNT
} Rank;

typedef enum {
  SUIT_CLUBS = 0,
  SUIT_DIAMONDS,
  SUIT_HEARTS,
  SUIT_SPADES,
  SUIT_COUNT
} Suit;

typedef struct {
  uint32_t card_id; // Unique identifier for the card, used in resetting the
                    // deck and making shuffles consistent
  uint8_t rank;
  uint8_t suit;
  uint8_t enhancement;
  uint8_t seal;
  uint8_t edition;
  uint16_t feature_id; // 4 bits rank, 2 bits suit, 4 bits enhancement, 3 bits
                       // seal, 3 bits edition
} Card;

typedef enum {
  HAND_HIGH_CARD = 0,
  HAND_PAIR,
  HAND_TWO_PAIR,
  HAND_THREE_OF_A_KIND,
  HAND_STRAIGHT,
  HAND_FLUSH,
  HAND_FULL_HOUSE,
  HAND_FOUR_OF_A_KIND,
  HAND_STRAIGHT_FLUSH,
  HAND_FIVE_OF_A_KIND,
  HAND_FLUSH_HOUSE,
  HAND_FLUSH_FIVE,
  HAND_TYPE_COUNT,
  HAND_TYPE_NONE = -1,
  HAND_TYPE_INVALID = -2
} HandTypes;

// Required that you have some struct for your env
// Recommended that you name it the same as the env file
typedef struct {
  Log log; // Required field. Env binding code uses this to aggregate logs
  float *observations; // Required. You can use any obs type, but make sure it
                       // matches in Python!
  float *actions;   // Required. int* for discrete/multidiscrete, float* for box
  float *rewards;   // Required
  float *terminals; // Required. We don't yet have truncations as standard yet
  int num_agents;   // Required
  unsigned int rng; // Required

  int tick; // Number of steps taken in the current episode, can be used for
            // time-based rewards or terminations instead of or in addition to
            // env-defined ones

  int seed;
  int run_seed;
  int rng_state;

  // ======= Deck Info =======
  Card cards[MAX_DECK_SIZE];
  uint8_t undrawn_cards[MAX_DECK_SIZE]; // Stores indices of cards currently in
                                        // the undrawn deck
  uint8_t num_undrawn_cards;

  uint8_t hand[MAX_HAND_SIZE]; // Stores indices of cards currently in hand
  uint8_t num_cards_in_hand;   // Note that this is different from hand size.
  // Hanged Man, Cryptid, and other cards that change deck size
  // can cause these to differ, while keeping hand_size constant.

  // TODO: Add selected_mask for selecting multiple cards at once
  uint8_t num_selected_cards;
  bool selected_mask[MAX_DECK_SIZE]; // Whether each card in hand is currently
                                     // selected by the player

  uint8_t discarded[MAX_DECK_SIZE]; // Stores indices of cards currently in the
                                    // discard pile (played or discarded cards).
                                    // Note that destroyed cards are not stored
                                    // here.
  uint8_t num_discarded_cards;

  uint8_t free_stack[MAX_DECK_SIZE]; // Stores indices of free slots in the
                                     // cards array. Used for cards that are
                                     // destroyed, not used for discards.
  uint8_t free_top;

  uint8_t next_card_id; // Total number of cards in the game, used for
                        // consistent shuffling and resetting

  uint8_t location_by_card[MAX_DECK_SIZE]; // For debugging purposes only.
                                           // 0 = None/Destroyed, 1 = Undrawn, 2
                                           // = Hand, 3 = Discard. Updated each
                                           // step in compute_observations for
                                           // easier observation encoding

  bool debuffed[MAX_DECK_SIZE]; // Whether each card in hand is currently
                                // debuffed by something like a boss blind
  // For Observations:
  uint8_t sorted_undrawn_cards
      [MAX_DECK_SIZE]; // Stores indices of undrawn cards, sorted by their
                       // feature_id for easier observation encoding. Updated
                       // each step in compute_observations.

  bool undrawn_cards_changed; // Whether the undrawn cards list has changed
                              // since last observation update
  bool hand_changed;      // Whether the hand has changed since last observation
                          // update
  bool discarded_changed; // Whether the discarded cards list has changed since
                          // last observation update

  // ======= Round Info =======
  uint8_t hand_size;
  uint8_t num_hands;
  uint8_t num_discards;

  uint8_t remaining_hands;
  uint8_t remaining_discards;

  double round_score;
  double required_score;

  uint8_t num_played_cards;
  uint8_t played_no_card_hands_counts;
  uint8_t played_hand_type_counts[HAND_TYPE_COUNT];

  uint8_t num_discarded_cards_log;
  uint8_t discarded_no_card_hands_counts;
  uint8_t discarded_hand_type_counts[HAND_TYPE_COUNT];

  // ====== Playthrough Info ======

  // ====== Sweepable Parameters ======
  float reward_completed_blind; // !! Must Set in .ini/during initialization !!
  float reward_unused_hand;     // !! Must Set in .ini/during initialization !!
  float reward_failure_penalty; // !! Must Set in .ini/during initialization !!
  //     -- Rewards/Penalties before action masking is implemented --
  float reward_illegal_action_penalty; // !! Must Set in .ini/during
                                       // initialization !!
  float reward_deselect_penalty; // !! Must Set in .ini/during initialization !!
} Balatro;

// Xorshift32 RNG for simple randomization needs
unsigned int rng_next(unsigned int *state) {
  unsigned int x = *state;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;

  // printf("RNG state: %u -> %u\n",  *state, x); // Debugging line to check RNG
  // state

  *state = x;
  return x;
}

void init_cards(Balatro *env) {
  memset(env->cards, 0, sizeof(env->cards));

  memset(env->undrawn_cards, 0, sizeof(env->undrawn_cards));
  memset(env->hand, 0, sizeof(env->hand));
  memset(env->selected_mask, 0, sizeof(env->selected_mask));
  memset(env->discarded, 0, sizeof(env->discarded));
  memset(env->free_stack, 0, sizeof(env->free_stack));

  memset(env->location_by_card, 0, sizeof(env->location_by_card));
  memset(env->debuffed, 0, sizeof(env->debuffed));
  memset(env->sorted_undrawn_cards, 0, sizeof(env->sorted_undrawn_cards));

  uint32_t id = 0;
  for (uint8_t rank = 0; rank < RANK_COUNT; rank++) {
    for (uint8_t suit = 0; suit < SUIT_COUNT; suit++) {
      env->cards[id] =
          (Card){.card_id = id,
                 .rank = rank,
                 .suit = suit,
                 .enhancement = 0,
                 .seal = 0,
                 .edition = 0,
                 .feature_id = (uint16_t)((rank << 12) | (suit << 10))};
      env->undrawn_cards[id] = id;

      env->location_by_card[id] = 1; // Undrawn

      id++;
    }
  }

  env->num_undrawn_cards = id;
  env->next_card_id = id;
  env->num_cards_in_hand = 0;
  env->num_discarded_cards = 0;
  env->num_selected_cards = 0;
  memset(env->selected_mask, 0, sizeof(env->selected_mask));

  env->free_top = 0;

  // Add all empty slots to the free stack
  for (uint16_t i = env->num_undrawn_cards; i < MAX_DECK_SIZE; i++) {
    env->free_stack[env->free_top++] = (uint8_t)i;
  }
}

void shuffle_and_reset_deck(Balatro *env, unsigned int *rng_state) {
  uint8_t n = env->num_undrawn_cards;

  // Move all cards from hand and discard pile back into deck
  for (uint8_t i = 0; i < env->num_cards_in_hand; i++) {
    env->undrawn_cards[n++] = env->hand[i];
  }
  for (uint8_t i = 0; i < env->num_discarded_cards; i++) {
    env->undrawn_cards[n++] = env->discarded[i];
  }

  env->num_undrawn_cards = n;
  env->num_cards_in_hand = 0;
  env->num_discarded_cards = 0;
  env->num_selected_cards = 0;

  memset(env->selected_mask, 0, sizeof(env->selected_mask));

  // Insertion sort to maintain a sorted deck before shuffling, for consistent
  // shuffling and resetting
  for (uint8_t i = 1; i < n; i++) {
    uint8_t key = env->undrawn_cards[i];
    uint32_t key_id = env->cards[key].card_id;

    uint8_t j = i;
    while (j > 0) {
      uint8_t prev = env->undrawn_cards[j - 1];
      if (env->cards[prev].card_id <= key_id)
        break;
      env->undrawn_cards[j] = prev;
      j--;
    }
    env->undrawn_cards[j] = key;
  }

  // Fisher-Yates shuffle
  for (uint8_t i = env->num_undrawn_cards - 1; i > 0; i--) {
    unsigned int rand_idx = rng_next(rng_state) % (i + 1);
    // Swap undrawn_cards[i] and undrawn_cards[rand_idx]
    uint8_t temp = env->undrawn_cards[i];
    env->undrawn_cards[i] = env->undrawn_cards[rand_idx];
    env->undrawn_cards[rand_idx] = temp;
  }

  env->undrawn_cards_changed = true;
  env->hand_changed = true;
  env->discarded_changed = true;
}

int draw_card(Balatro *env) {
  if (env->num_undrawn_cards == 0) {
    return -1; // No cards left to draw
  }
  if (env->num_cards_in_hand >= MAX_HAND_SIZE) {
    return -2; // Hand is full
  }

  uint8_t card_idx = env->undrawn_cards[--env->num_undrawn_cards];
  env->hand[env->num_cards_in_hand++] = card_idx;

  env->undrawn_cards_changed = true;
  env->hand_changed = true;

  return (int)card_idx;
}

// TODO: Refactor to use selected_mask for selecting multiple cards at once,
int discard_card(Balatro *env, uint8_t hand_idx) {
  if (hand_idx >= env->num_cards_in_hand)
    return -1;
  if (env->num_discarded_cards >= MAX_DECK_SIZE)
    return -2; // Discard pile is full

  uint8_t card_idx = env->hand[hand_idx];
  env->discarded[env->num_discarded_cards++] = card_idx;

  // Shift remaining cards in hand to fill the gap
  for (uint8_t i = hand_idx; i < env->num_cards_in_hand - 1; i++) {
    env->hand[i] = env->hand[i + 1];
  }
  env->num_cards_in_hand--;

  env->hand_changed = true;
  env->discarded_changed = true;

  return (int)card_idx;
}

// Uses env->selected_mask for discarding respective cards
int discard_cards(Balatro *env) {
  if (env->num_selected_cards == 0)
    return 0;
  if (env->num_discarded_cards + env->num_selected_cards > MAX_DECK_SIZE)
    return -2; // Discard pile is full

  uint8_t num_discarded_cards = 0;
  uint8_t write_hand_idx = 0;
  uint8_t original_num_cards_in_hand = env->num_cards_in_hand;

  for (uint8_t hand_idx = 0; hand_idx < original_num_cards_in_hand;
       hand_idx += 1) {
    uint8_t card_idx = env->hand[hand_idx];

    if (!env->selected_mask[card_idx]) {
      env->hand[write_hand_idx++] = env->hand[hand_idx];
    } else {
      env->discarded[env->num_discarded_cards++] = card_idx;
      env->selected_mask[card_idx] = false;
      num_discarded_cards++;
    }
  }

  env->hand_changed = true;
  env->discarded_changed = true;

  env->num_cards_in_hand -= num_discarded_cards;
  env->num_selected_cards -= num_discarded_cards;
  // TODO: once functionality is confirmed, consider using the below
  // implementation since it'll be stronger. env->num_cards_in_hand =
  // write_hand_idx; env->num_selected_cards = 0;

  // Sanity check; TODO: Delete once functionality is confirmed
  if (env->num_selected_cards != 0 || env->num_cards_in_hand != write_hand_idx)
    return -1; // Discard mismatch failure

  return num_discarded_cards;
}

int destroy_card(Balatro *env, uint8_t hand_idx) {
  if (hand_idx >= env->num_cards_in_hand) {
    return -1; // Invalid hand index
  }
  if (env->free_top >= MAX_DECK_SIZE) {
    return -2; // Free stack is full, should never happen if MAX_DECK_SIZE is
               // large enough
  }

  uint8_t card_idx = env->hand[hand_idx];
  env->free_stack[env->free_top++] = card_idx;

  // Shift remaining cards in hand to fill the gap
  env->hand[hand_idx] =
      env->hand[env->num_cards_in_hand - 1]; // Move last card to the gap
  env->num_cards_in_hand--;

  env->hand_changed = true;

  return (int)card_idx;
}

int create_card(Balatro *env, uint8_t rank, uint8_t suit) {
  if (env->free_top == 0) {
    return -1; // No free slots to create a new card
  }
  if (env->num_cards_in_hand >= MAX_HAND_SIZE) {
    return -2; // Hand is full
  }

  uint8_t card_idx = env->free_stack[--env->free_top];
  env->cards[card_idx] =
      (Card){.card_id = env->next_card_id++,
             .rank = rank,
             .suit = suit,
             .enhancement = 0,
             .seal = 0,
             .edition = 0,
             .feature_id = (uint16_t)((rank << 12) | (suit << 10))};
  env->hand[env->num_cards_in_hand++] = card_idx;

  env->hand_changed = true;

  return (int)card_idx;
}

/*
 * Sorts in place the cards in idxs by feature_id (Rank, Suit, Enhancement,
 * etc.) using insertion sort.
 *
 * cards - cards array directly from env
 * idxs  - list of indices of cards within the card stack.
 * n     - Number of total cards to sort.
 *
 * Sorted by (rank, suit, enhancement, seal, edition, location, debuffed)
 * instead of just card ID for better generalization to new cards These are the
 * ranges of the different features
 *  - Rank (15): 0 = None, otherwise card->rank + 1; if card->enhancement =
 * stone, then rank = 14 (Will be implemented later, don't do now)
 *  - Suit (5): 0 = None, otherwise card->suit + 1
 *  - Enhancement (9): 0 - 8, copy directly from the card struct
 *  - Seal (5): 0 - 4, copy directly from the card struct
 *  - Edition (5): 0 - 4, copy directly from the card struct
 *  - Location (4): 0 = None, 1 = Undrawn, 2 = Hand, 3 = Discard
 *  - Debuffed (2): 0 = No, 1 = Yes (For cards that are currently debuffed by
 * something like a boss blind)
 *  - For each card, we have 3 bytes of observations:
 *    - Byte 1: Rank * 5 + Suit
 *    - Byte 2: Edition + Seal * 5 + Enhancement * 25
 *    - Byte 3: Location + Debuffed * 4
 */
static inline void sort_indices_by_feature(Card *cards, uint8_t *idxs,
                                           uint8_t n) {
  for (uint8_t i = 1; i < n; i++) {
    uint8_t key = idxs[i];
    uint16_t key_f = cards[key].feature_id;

    uint8_t j = i;
    while (j > 0) {
      uint8_t prev = idxs[j - 1];
      if (cards[prev].feature_id <= key_f)
        break;
      idxs[j] = prev;
      j--;
    }
    idxs[j] = key;
  }
}

double calculate_hand_score(Balatro *env) {
  if (env->num_selected_cards == 0)
    return -1.0; // No cards selected
  uint8_t selected_card_indices[MAX_SELECTION_SIZE];
  uint8_t rank_counts[RANK_COUNT] = {0};
  uint8_t suit_counts[SUIT_COUNT] = {0};

  uint8_t num_selected_cards_processed = 0;

  for (int i = 0; i < env->num_cards_in_hand; i++) {
    uint8_t card_idx = env->hand[i];

    if (!env->selected_mask[card_idx])
      continue;

    Card card = env->cards[card_idx];
    rank_counts[card.rank]++;
    suit_counts[card.suit]++;

    selected_card_indices[num_selected_cards_processed] = card_idx;
    num_selected_cards_processed++;
  }

  if (num_selected_cards_processed != env->num_selected_cards)
    return -2.0; // Selected mask doesn't contain the same number of selected
                 // cards;

  uint8_t pairs = 0;
  bool three_of_a_kind = false;
  bool four_of_a_kind = false;
  bool five_of_a_kind = false;
  bool flush = false;
  bool straight = false;
  bool no_duplicates = true;
  uint16_t rank_mask = 0;
  for (int r = 0; r < RANK_COUNT; r++) {
    uint8_t count = rank_counts[r];
    if (count == 0) {
      continue;
    }

    rank_mask |= (uint16_t)(1u << r);
    switch (count) {
    case 5:
      five_of_a_kind = true;
      break;
    case 4:
      four_of_a_kind = true;
      break;
    case 3:
      three_of_a_kind = true;
      break;
    case 2:
      pairs++;
      break;
    case 1:
      continue;
      break;
    }
    no_duplicates =
        false; // Only count=1 has a continue, skipping over this line
  }

  for (int s = 0; s < SUIT_COUNT; s++) {
    if (suit_counts[s] >= 5) {
      flush = true;
      break;
    }
  }

  // Assumes stuff can't occur with power-ups that affect ways to make straight
  if (no_duplicates) {
    for (uint8_t start = 0; start <= 8; start++) {
      if (((rank_mask >> start) & 0x1F) == 0x1F) {
        straight = true;
        break;
      }
    }

    // Check if selected cards
    // is containing A--2--3--4--5 straight
    //    i.e. bits 12, 0, 1, 2, 3
    if (!straight) {
      straight = (rank_mask | ((1u << 12) | 0x0F)) == rank_mask;
    }
  }

  HandTypes played_hand;
  if (five_of_a_kind && flush)
    played_hand = HAND_FLUSH_FIVE;
  else if (three_of_a_kind && pairs == 1 && flush)
    played_hand = HAND_FLUSH_HOUSE;
  else if (five_of_a_kind)
    played_hand = HAND_FIVE_OF_A_KIND;
  else if (straight && flush)
    played_hand = HAND_STRAIGHT_FLUSH;
  else if (four_of_a_kind)
    played_hand = HAND_FOUR_OF_A_KIND;
  else if (three_of_a_kind && pairs == 1)
    played_hand = HAND_FULL_HOUSE;
  else if (flush)
    played_hand = HAND_FLUSH;
  else if (straight)
    played_hand = HAND_STRAIGHT;
  else if (three_of_a_kind)
    played_hand = HAND_THREE_OF_A_KIND;
  else if (pairs == 2)
    played_hand = HAND_TWO_PAIR;
  else if (pairs == 1)
    played_hand = HAND_PAIR;
  else
    played_hand = HAND_HIGH_CARD;

  double chips = 0.0;
  double mult = 0.0;

  switch (played_hand) {
  case HAND_FLUSH_FIVE:
    chips = 160.0;
    mult = 16.0;
    break;
  case HAND_FLUSH_HOUSE:
    chips = 140.0;
    mult = 14.0;
    break;
  case HAND_FIVE_OF_A_KIND:
    chips = 120.0;
    mult = 12.0;
    break;
  case HAND_STRAIGHT_FLUSH:
    chips = 100.0;
    mult = 8.0;
    break;
  case HAND_FOUR_OF_A_KIND:
    chips = 60.0;
    mult = 7.0;
    break;
  case HAND_FULL_HOUSE:
    chips = 40.0;
    mult = 4.0;
    break;
  case HAND_FLUSH:
    chips = 35.0;
    mult = 4.0;
    break;
  case HAND_STRAIGHT:
    chips = 30.0;
    mult = 4.0;
    break;
  case HAND_THREE_OF_A_KIND:
    chips = 30.0;
    mult = 3.0;
    break;
  case HAND_TWO_PAIR:
    chips = 20.0;
    mult = 2.0;
    break;
  case HAND_PAIR:
    chips = 10.0;
    mult = 2.0;
    break;
  case HAND_HIGH_CARD:
    chips = 5.0;
    mult = 1.0;
    break;
  }

  // score each card individually
  // TODO: Figure out if ordering matters or get better results forcing optimal
  // score every time Only matters for glass cards and reordering xmult vs
  // +mult; Less relevant for jokerless since you wouldn't normally play glass
  // cards if not trying to win.
  for (uint8_t i = 0; i < env->num_selected_cards; i++) {
    uint8_t card_idx = selected_card_indices[i];
    Card card = env->cards[card_idx];
    switch (card.rank) {
    case RANK_ACE:
      chips += 11;
      break;
    case RANK_KING:
    case RANK_QUEEN:
    case RANK_JACK:
      chips += 10;
      break;
    default:
      chips += card.rank + 2; // Rank 2 is 0, but gives 2 chips.
      break;
    }
  }

  return chips * mult;
}

HandTypes get_hand_type(Balatro *env) {
  if (env->num_selected_cards == 0)
    return HAND_TYPE_NONE; // No cards selected
  uint8_t selected_card_indices[MAX_SELECTION_SIZE];
  uint8_t rank_counts[RANK_COUNT] = {0};
  uint8_t suit_counts[SUIT_COUNT] = {0};

  uint8_t num_selected_cards_processed = 0;

  for (int i = 0; i < env->num_cards_in_hand; i++) {
    uint8_t card_idx = env->hand[i];

    if (!env->selected_mask[card_idx])
      continue;

    Card card = env->cards[card_idx];
    rank_counts[card.rank]++;
    suit_counts[card.suit]++;

    selected_card_indices[num_selected_cards_processed] = card_idx;
    num_selected_cards_processed++;
  }

  if (num_selected_cards_processed != env->num_selected_cards)
    return HAND_TYPE_INVALID; // Selected mask doesn't contain the same number
                              // of selected cards;

  uint8_t pairs = 0;
  bool three_of_a_kind = false;
  bool four_of_a_kind = false;
  bool five_of_a_kind = false;
  bool flush = false;
  bool straight = false;
  bool no_duplicates = true;
  uint16_t rank_mask = 0;
  for (int r = 0; r < RANK_COUNT; r++) {
    uint8_t count = rank_counts[r];
    if (count == 0) {
      continue;
    }

    rank_mask |= (uint16_t)(1u << r);
    switch (count) {
    case 5:
      five_of_a_kind = true;
      break;
    case 4:
      four_of_a_kind = true;
      break;
    case 3:
      three_of_a_kind = true;
      break;
    case 2:
      pairs++;
      break;
    case 1:
      continue;
      break;
    }
    no_duplicates =
        false; // Only count=1 has a continue, skipping over this line
  }

  for (int s = 0; s < SUIT_COUNT; s++) {
    if (suit_counts[s] >= 5) {
      flush = true;
      break;
    }
  }

  // Assumes stuff can't occur with power-ups that affect ways to make straight
  if (no_duplicates) {
    for (uint8_t start = 0; start <= 8; start++) {
      if (((rank_mask >> start) & 0x1F) == 0x1F) {
        straight = true;
        break;
      }
    }

    // Check if selected cards
    // is containing A--2--3--4--5 straight
    //    i.e. bits 12, 0, 1, 2, 3
    if (!straight) {
      straight = (rank_mask | ((1u << 12) | 0x0F)) == rank_mask;
    }
  }

  HandTypes played_hand;
  if (five_of_a_kind && flush)
    played_hand = HAND_FLUSH_FIVE;
  else if (three_of_a_kind && pairs == 1 && flush)
    played_hand = HAND_FLUSH_HOUSE;
  else if (five_of_a_kind)
    played_hand = HAND_FIVE_OF_A_KIND;
  else if (straight && flush)
    played_hand = HAND_STRAIGHT_FLUSH;
  else if (four_of_a_kind)
    played_hand = HAND_FOUR_OF_A_KIND;
  else if (three_of_a_kind && pairs == 1)
    played_hand = HAND_FULL_HOUSE;
  else if (flush)
    played_hand = HAND_FLUSH;
  else if (straight)
    played_hand = HAND_STRAIGHT;
  else if (three_of_a_kind)
    played_hand = HAND_THREE_OF_A_KIND;
  else if (pairs == 2)
    played_hand = HAND_TWO_PAIR;
  else if (pairs == 1)
    played_hand = HAND_PAIR;
  else
    played_hand = HAND_HIGH_CARD;

  return played_hand;
}

void print_card(const Card *c) {
  const char *rank_str[] = {"2", "3",  "4", "5", "6", "7", "8",
                            "9", "10", "J", "Q", "K", "A"};

  const char *suit_str[] = {"♣ Clubs", "♦ Diamonds", "♥ Hearts", "♠ Spades"};

  if (c->rank >= RANK_COUNT || c->suit >= SUIT_COUNT) {
    printf("Invalid card\n");
    return;
  }

  printf("%s of %s\n", rank_str[c->rank], suit_str[c->suit]);
}

void verify_observations(Balatro *env, const char *prefix_string) {
  const uint16_t OBS_SIZE = NUM_UNDRAWN_OBS * MAX_DECK_SIZE +
                            NUM_HAND_OBS * MAX_HAND_SIZE +
                            NUM_DISCARD_OBS * MAX_DECK_SIZE + 6;

  for (int i = 0; i < OBS_SIZE; i++) {
    if (!isfinite(env->observations[i])) {
      printf("<%s> Bad obs at %d: %f [0x%x]\n", prefix_string, i,
             env->observations[i], *(uint32_t *)&(env->observations[i]));
      abort();
    }
  }

  uint16_t verify_idx = 0;

  for (int i = 0; i < MAX_DECK_SIZE; i++) {
    for (int j = 0; j < NUM_UNDRAWN_OBS; j++) {
      if (i < env->num_undrawn_cards && (env->observations[verify_idx] < -0.1 ||
                                         env->observations[verify_idx] > 1.1)) {
        printf("<%s> (Tick #%d - Undrawn) Observation outside [0, 1] at idx %u "
               "(Undrawn card #%d, field #%d; Currently %f [0x%x].)\n",
               prefix_string, env->tick, verify_idx, i, j,
               env->observations[verify_idx],
               *(uint32_t *)&(env->observations[verify_idx]));
      } else if (i >= env->num_undrawn_cards &&
                 fabsf(env->observations[verify_idx]) > 1e-8f) {
        printf("<%s> (Tick #%d - Undrawn) Empty card has nonzero obs at idx %u "
               "(Undrawn card #%d, field #%d; Currently %f [0x%x]) [%u "
               "undrawn, %u in hand, %u discarded]\n",
               prefix_string, env->tick, verify_idx, i, j,
               env->observations[verify_idx],
               *(uint32_t *)&(env->observations[verify_idx]),
               env->num_undrawn_cards, env->num_cards_in_hand,
               env->num_discarded_cards);
      }
      verify_idx++;
    }
  }

  for (int i = 0; i < MAX_HAND_SIZE; i++) {
    for (int j = 0; j < NUM_HAND_OBS; j++) {
      if (i < env->num_cards_in_hand && (env->observations[verify_idx] < -0.1 ||
                                         env->observations[verify_idx] > 1.1)) {
        printf("<%s> (Tick #%d - Hand) Observation outside [0, 1] at idx %u "
               "(Hand card #%d, field #%d; Currently %f [0x%x].)\n",
               prefix_string, env->tick, verify_idx, i, j,
               env->observations[verify_idx],
               *(uint32_t *)&(env->observations[verify_idx]));
      } else if (i >= env->num_cards_in_hand &&
                 fabsf(env->observations[verify_idx]) > 1e-8f) {
        printf("<%s> (Tick #%d - Hand) Empty card has nonzero obs at idx %u "
               "(Hand card #%d, field #%d; Currently %f [0x%x]) [%u undrawn, "
               "%u in hand, %u discarded]\n",
               prefix_string, env->tick, verify_idx, i, j,
               env->observations[verify_idx],
               *(uint32_t *)&(env->observations[verify_idx]),
               env->num_undrawn_cards, env->num_cards_in_hand,
               env->num_discarded_cards);
      }
      verify_idx++;
    }
  }

  for (int i = 0; i < MAX_DECK_SIZE; i++) {
    for (int j = 0; j < NUM_DISCARD_OBS; j++) {
      if (i < env->num_discarded_cards &&
          (env->observations[verify_idx] < -0.1 ||
           env->observations[verify_idx] > 1.1)) {
        printf("<%s> (Tick #%d - Discard) Observation outside [0, 1] at idx %u "
               "(Discard card #%d, field #%d; Currently %f [0x%x].)\n",
               prefix_string, env->tick, verify_idx, i, j,
               env->observations[verify_idx],
               *(uint32_t *)&(env->observations[verify_idx]));
      } else if (i >= env->num_discarded_cards &&
                 fabsf(env->observations[verify_idx]) > 1e-8f) {
        printf("<%s> (Tick #%d - Discard) Empty card has nonzero obs at idx %u "
               "(Discard card #%d, field #%d; Currently %f [0x%x]) [%u "
               "undrawn, %u in hand, %u discarded]\n",
               prefix_string, env->tick, verify_idx, i, j,
               env->observations[verify_idx],
               *(uint32_t *)&(env->observations[verify_idx]),
               env->num_undrawn_cards, env->num_cards_in_hand,
               env->num_discarded_cards);
      }
      verify_idx++;
    }
  }
}

void compute_observations(Balatro *env) {
  // TODO: Find better way to encode magic number?
  // const uint32_t OBS_SIZE = NUM_UNDRAWN_OBS * MAX_DECK_SIZE + NUM_HAND_OBS *
  // MAX_HAND_SIZE + NUM_DISCARD_OBS * MAX_DECK_SIZE + 6;
  const uint32_t OBS_SIZE = 1611;
  if (env->undrawn_cards_changed) {
    for (uint8_t i = 0; i < env->num_undrawn_cards; i++) {
      env->sorted_undrawn_cards[i] = env->undrawn_cards[i];
    }
    sort_indices_by_feature(env->cards, env->sorted_undrawn_cards,
                            env->num_undrawn_cards);
    env->undrawn_cards_changed = false;
  }

  if (env->hand_changed) {
    sort_indices_by_feature(env->cards, env->hand, env->num_cards_in_hand);
    env->hand_changed = false;
  }

  if (env->discarded_changed) {
    sort_indices_by_feature(env->cards, env->discarded,
                            env->num_discarded_cards);
    env->discarded_changed = false;
  }

  memset(env->observations, 0, OBS_SIZE * sizeof(float));
  // verify_observations(env, "after memset 0");

  uint16_t obs_idx = 0;
  uint16_t obs_checkpoint = 0; // Used for marking starts of blocks of related
                               // obs (e.g. deck, hand, or discard piles)

  // Encode undrawn cards
  for (uint16_t i = 0; i < env->num_undrawn_cards; i++) {
    uint8_t card_idx = env->sorted_undrawn_cards[i];
    Card card = env->cards[card_idx];
    env->observations[obs_idx++] =
        (float)((card.rank + 1) * 5 + (card.suit + 1)) /
        ((RANK_COUNT + 1) *
         (SUIT_COUNT +
          1)); // Rank and Suit ard incremented by 1 to make 0 represent "None"
    env->observations[obs_idx++] =
        card.edition + card.seal * 5 + card.enhancement * 25;
    env->observations[obs_idx++] = 0.25f; // Location: Undrawn
  }
  // verify_observations(env, "after undrawn");
  obs_checkpoint += NUM_UNDRAWN_OBS * MAX_DECK_SIZE;
  obs_idx = obs_checkpoint;

  // Encode cards in hand
  for (uint16_t i = 0; i < env->num_cards_in_hand; i++) {
    uint8_t card_idx = env->hand[i];
    Card card = env->cards[card_idx];
    env->observations[obs_idx++] =
        (float)((card.rank + 1) * 5 + (card.suit + 1)) /
        ((RANK_COUNT + 1) *
         (SUIT_COUNT +
          1)); // Rank and Suit ard incremented by 1 to make 0 represent "None"
    env->observations[obs_idx++] =
        card.edition + card.seal * 5 + card.enhancement * 25;
    env->observations[obs_idx++] = 0.5f; // Location: Hand
    env->observations[obs_idx++] =
        env->selected_mask[card_idx];                       // Selected Flag
    env->observations[obs_idx++] = env->debuffed[card_idx]; // Debuffed Flag
  }
  // verify_observations(env, "after hand");
  obs_checkpoint += NUM_HAND_OBS * MAX_HAND_SIZE;
  obs_idx = obs_checkpoint;

  // Encode discarded cards
  for (uint16_t i = 0; i < env->num_discarded_cards; i++) {
    uint8_t card_idx = env->discarded[i];
    Card card = env->cards[card_idx];
    env->observations[obs_idx++] =
        (float)((card.rank + 1) * 5 + (card.suit + 1)) /
        ((RANK_COUNT + 1) *
         (SUIT_COUNT +
          1)); // Rank and Suit ard incremented by 1 to make 0 represent "None"
    env->observations[obs_idx++] =
        card.edition + card.seal * 5 + card.enhancement * 25;
    env->observations[obs_idx++] = 0.75f; // Location: Discard
  }
  // verify_observations(env, "after discard");
  obs_idx = NUM_UNDRAWN_OBS * MAX_DECK_SIZE + NUM_HAND_OBS * MAX_HAND_SIZE +
            NUM_DISCARD_OBS * MAX_DECK_SIZE;
  obs_checkpoint += NUM_DISCARD_OBS * MAX_DECK_SIZE;
  obs_idx = obs_checkpoint;

  // TODO: Figure out good way to encode hand size, number of hands, number of
  // discards.
  env->observations[obs_idx++] =
      (float)env->hand_size / MAX_HAND_SIZE; // Hand size
  env->observations[obs_idx++] =
      (float)env->num_hands / STARTING_NUM_HANDS; // Number of hands
  env->observations[obs_idx++] =
      (float)env->num_discards / STARTING_NUM_DISCARDS; // Number of discards

  env->observations[obs_idx++] = (float)env->remaining_hands /
                                 env->num_hands; // Fraction of remaining hands
  env->observations[obs_idx++] =
      (float)env->remaining_discards /
      env->num_discards; // Fraction of remaining discards

  env->observations[obs_idx++] =
      env->round_score / env->required_score; // Fraction of score remaining

  // ---------- Verify Observations ----------
  // verify_observations(env, "After everything");
}

void add_log(Balatro *env) {
  const float BLIND_DEFEAT_WEIGHT = 0.8;
  float fractional_score_perf =
      MIN(1.0, (env->round_score / env->required_score));
  env->log.perf += fractional_score_perf * BLIND_DEFEAT_WEIGHT;
  if (env->round_score >= env->required_score) {
    float unused_hands_proportion =
        (float)env->remaining_hands /
        MAX(1, env->num_hands - 1); // Subtract by 1 since you need to use at
                                    // least the first hand
    env->log.perf += unused_hands_proportion * (1 - BLIND_DEFEAT_WEIGHT);
    env->log.win += 1;
  }

  env->log.score += env->round_score;
  env->log.episode_length += env->tick;
  env->log.episode_return += env->round_score;
  env->log.remaining_hands += env->remaining_hands;
  env->log.remaining_discards += env->remaining_discards;
  env->log.num_played_cards += env->num_played_cards;
  env->log.num_discarded_cards += env->num_discarded_cards_log;

  env->log.played_hand_type_none += env->played_no_card_hands_counts;
  env->log.played_hand_type_high_card +=
      env->played_hand_type_counts[HAND_HIGH_CARD];
  env->log.played_hand_type_pair += env->played_hand_type_counts[HAND_PAIR];
  env->log.played_hand_type_two_pair +=
      env->played_hand_type_counts[HAND_TWO_PAIR];
  env->log.played_hand_type_three_of_a_kind +=
      env->played_hand_type_counts[HAND_THREE_OF_A_KIND];
  env->log.played_hand_type_straight +=
      env->played_hand_type_counts[HAND_STRAIGHT];
  env->log.played_hand_type_flush += env->played_hand_type_counts[HAND_FLUSH];
  env->log.played_hand_type_full_house +=
      env->played_hand_type_counts[HAND_FULL_HOUSE];
  env->log.played_hand_type_four_of_a_kind +=
      env->played_hand_type_counts[HAND_FOUR_OF_A_KIND];
  env->log.played_hand_type_straight_flush +=
      env->played_hand_type_counts[HAND_STRAIGHT_FLUSH];
  env->log.played_hand_type_five_of_a_kind +=
      env->played_hand_type_counts[HAND_FIVE_OF_A_KIND];
  env->log.played_hand_type_flush_house +=
      env->played_hand_type_counts[HAND_FLUSH_HOUSE];
  env->log.played_hand_type_flush_five +=
      env->played_hand_type_counts[HAND_FLUSH_FIVE];

  env->log.discarded_hand_type_none += env->discarded_no_card_hands_counts;
  env->log.discarded_hand_type_high_card +=
      env->discarded_hand_type_counts[HAND_HIGH_CARD];
  env->log.discarded_hand_type_pair +=
      env->discarded_hand_type_counts[HAND_PAIR];
  env->log.discarded_hand_type_two_pair +=
      env->discarded_hand_type_counts[HAND_TWO_PAIR];
  env->log.discarded_hand_type_three_of_a_kind +=
      env->discarded_hand_type_counts[HAND_THREE_OF_A_KIND];
  env->log.discarded_hand_type_straight +=
      env->discarded_hand_type_counts[HAND_STRAIGHT];
  env->log.discarded_hand_type_flush +=
      env->discarded_hand_type_counts[HAND_FLUSH];
  env->log.discarded_hand_type_full_house +=
      env->discarded_hand_type_counts[HAND_FULL_HOUSE];
  env->log.discarded_hand_type_four_of_a_kind +=
      env->discarded_hand_type_counts[HAND_FOUR_OF_A_KIND];
  env->log.discarded_hand_type_straight_flush +=
      env->discarded_hand_type_counts[HAND_STRAIGHT_FLUSH];
  env->log.discarded_hand_type_five_of_a_kind +=
      env->discarded_hand_type_counts[HAND_FIVE_OF_A_KIND];
  env->log.discarded_hand_type_flush_house +=
      env->discarded_hand_type_counts[HAND_FLUSH_HOUSE];
  env->log.discarded_hand_type_flush_five +=
      env->discarded_hand_type_counts[HAND_FLUSH_FIVE];

  env->log.n++;
}

void init(Balatro *env) {
  if (env->seed == 0) {
    srand(time(NULL));
  } else {
    srand(env->seed);
  }
}

// Required function
void c_reset(Balatro *env) {
  // int tiles = env->size*env->size;
  // memset(env->observations, 0, tiles*sizeof(unsigned char));
  // env->observations[tiles/2] = AGENT;
  // env->r = env->size/2;
  // env->c = env->size/2;
  // env->tick = 0;
  // int target_idx;
  // do {
  //     target_idx = rand() % tiles;
  // } while (target_idx == tiles/2);
  // env->observations[target_idx] = TARGET;

  // AAAAAAAAAAAAAAAAAAAAAAAAA !!!!!!!!!!!!!!!!!! TODO TODO TODO RESET
  // EVERYTHING
  env->run_seed = rand();
  env->rng_state = env->run_seed;
  // printf("Initial RNG state: %u\n", env->rng_state); // Debugging line to
  // check initial RNG state

  env->tick = 0;
  env->hand_size = STARTING_HAND_SIZE;
  env->num_hands = STARTING_NUM_HANDS;
  env->num_discards = STARTING_NUM_DISCARDS;
  env->round_score = 0;

  env->num_undrawn_cards = 0;
  env->num_cards_in_hand = 0;
  env->num_discarded_cards = 0;
  env->num_selected_cards = 0;
  env->num_played_cards = 0;
  env->played_no_card_hands_counts = 0;
  memset(env->played_hand_type_counts, 0, sizeof(env->played_hand_type_counts));
  env->num_discarded_cards_log = 0;
  env->discarded_no_card_hands_counts = 0;
  memset(env->discarded_hand_type_counts, 0,
         sizeof(env->discarded_hand_type_counts));
  memset(env->selected_mask, 0, sizeof(env->selected_mask));

  memset(env->debuffed, 0, sizeof(env->debuffed));

  env->required_score = TEMP_TARGET_SCORE;

  init_cards(env);
  shuffle_and_reset_deck(env, (unsigned int *)&env->rng_state);

  for (uint8_t i = 0; i < env->hand_size; i++) {
    draw_card(env);
  }

  env->remaining_hands = env->num_hands;
  env->remaining_discards = env->num_discards;

  compute_observations(env);
}

// Required function
void c_step(Balatro *env) {
  env->tick += 1;

  int action = env->actions[0];
  env->terminals[0] = 0;
  env->rewards[0] = 0;

  // if (action == 6) {
  //     for (uint8_t i = 0; i < env->num_cards_in_hand; i++) {
  //         uint8_t card_idx = env->hand[i];
  //         env->rewards[0] += (float)env->cards[card_idx].rank /
  //         (env->num_cards_in_hand * RANK_COUNT); // Reward for keeping higher
  //         ranked cards in hand, to encourage the agent to learn to discard
  //         lower ranked cards first env->score += env->cards[card_idx].rank;
  //         // printf("Card #%u: %u\n", i, env->cards[card_idx].rank);
  //     }
  //     // printf("%f\n", env->rewards[0]);
  //     // printf("\n");
  //     for (uint8_t i = 0; i < 5; i++) {
  //         discard_card(env, i);
  //         draw_card(env);
  //     }
  //     env->remaining_hands--;
  //     if (env->remaining_hands == 0) {
  //
  //         add_log(env);
  //         env->terminals[0] = 1;
  //
  //         // printf("Final Score: %u \n\n\n", env->score);
  //         // printf("Final Score: %u\n", env->score);
  //         c_reset(env);
  //         return;
  //     }
  // } else if (action > 0 && action <= 5) {
  //     // if (env->remaining_discards != 0) {
  //     //     printf("discard, %u, %u, %u, (%u)\n", action,
  //     env->num_cards_in_hand, env->remaining_discards,
  //     env->cards[env->hand[action - 1]].rank);
  //     // }
  //     if (action <= 0 || action > env->num_cards_in_hand ||
  //     env->remaining_discards == 0) {
  //         // Invalid action, treat as noop
  //         return;
  //     }
  //     // uint8_t selected_card_idx = env->hand[action - 1];
  //     discard_card(env, action - 1);
  //     draw_card(env);
  //     env->remaining_discards--;
  // }

  /*
  ========================
  |   Action Breakdown   |
  ========================

  0 - NOOP (Only for Debugging Purposes


  ~~~~~~~~~~~~~~ IN ROUND ~~~~~~~~~~~~~~~
   - 1-16  - select/deselect card at that slot (15 total for MAX_HAND_SIZE)
   - 17    - Play Hand
   - 18    - Discard Hand

  ~~~~~~~~~~~~~~~~ SHOP ~~~~~~~~~~~~~~~~~
  TODO: Add more info about other actions

  */
  if (action == NOOP) {
    env->rewards[0] = env->reward_illegal_action_penalty;
    // TODO: Mask NOOP action; only left for debugging purposes & before action
    // mask implemented
  } else if (SELECT_FIRST_CARD <= action &&
             action < SELECT_FIRST_CARD + env->num_cards_in_hand) {
    uint8_t hand_idx = action - SELECT_FIRST_CARD;

    // TODO: Action Masking; remove once action masking implemented
    if (hand_idx >= env->num_cards_in_hand) {
      env->rewards[0] = env->reward_illegal_action_penalty;
    } else {
      uint8_t selected_card_idx = env->hand[hand_idx];
      // hand should be sorted because compute_observations
      // at the end of previous time step should've ensured that all stacks were
      // sorted

      if (!env->selected_mask[selected_card_idx]) {
        if (env->num_selected_cards >= MAX_SELECTION_SIZE) {
          env->rewards[0] += env->reward_illegal_action_penalty;
        } else {
          env->selected_mask[selected_card_idx] = true;
          env->num_selected_cards++;
        }
      } else {
        // Deselecting card
        // TODO: Action Masking - Consider masking/disallowing deselection
        // completely?

        // TODO DESELECTING IS NOT ALLOWED BUT UNCOMMENT TO ALLOW AGAIN
        // env->selected_mask[selected_card_idx] = false;
        // env->num_selected_cards--;
        env->rewards[0] += env->reward_deselect_penalty;
      }
    }
  } else if (SELECT_FIRST_CARD + env->num_cards_in_hand <= action &&
             action < SELECT_FIRST_CARD + MAX_HAND_SIZE) {
    env->rewards[0] = env->reward_illegal_action_penalty;
  } else if (action == PLAY_HAND) {
    if (env->num_selected_cards == 0) {
      env->rewards[0] = env->reward_illegal_action_penalty;
    }

    HandTypes played_hand_type = get_hand_type(env);
    if (played_hand_type == HAND_TYPE_NONE) {
      env->played_no_card_hands_counts++;
    } else if (played_hand_type != HAND_TYPE_INVALID) {
      env->played_hand_type_counts[played_hand_type]++;
    }

    double hand_score = calculate_hand_score(env);
    // printf("Scored %f\n", hand_score);
    if (hand_score == -1.0) {
      env->rewards[0] = env->reward_illegal_action_penalty;
    } else if (hand_score == -2.0) {
      printf("Error: selected mask doesn't contain same number of selected "
             "cards as `env->num_selected_cards`\n");
    } else {
      env->round_score += hand_score;
      // printf("Played hand with score %.2f, Total round score: %.2f, %u hands
      // left (including this one)\n", hand_score, env->round_score,
      // env->remaining_hands);
      uint8_t num_discarded_cards = discard_cards(env);
      env->num_played_cards += num_discarded_cards;
      for (uint8_t i = 0; i < num_discarded_cards; i++) {
        draw_card(env);
      }

      env->rewards[0] += MIN(1.0, hand_score / env->required_score) *
                         (1 - env->reward_completed_blind);
      env->remaining_hands--;

      if (env->round_score >= env->required_score) {
        env->rewards[0] += (float)env->reward_completed_blind *
                           (1 + env->remaining_hands) / env->num_hands;
        env->terminals[0] = 1;
        add_log(env);
        c_reset(env);
        return;
      } else if (env->remaining_hands == 0) {
        env->rewards[0] += env->reward_failure_penalty;
        env->terminals[0] = 1;
        add_log(env);
        c_reset(env);
        return;
      }
    }
  } else if (action == DISCARD) {
    HandTypes discarded_hand_type = get_hand_type(env);
    if (discarded_hand_type == HAND_TYPE_NONE) {
      env->discarded_no_card_hands_counts++;
    } else if (discarded_hand_type != HAND_TYPE_INVALID) {
      env->discarded_hand_type_counts[discarded_hand_type]++;
    }

    if (env->num_selected_cards == 0 || env->remaining_discards == 0) {
      env->rewards[0] = env->reward_illegal_action_penalty;
    } else {
      uint8_t num_discarded_cards = discard_cards(env);
      env->num_discarded_cards_log += num_discarded_cards;
      for (uint8_t i = 0; i < num_discarded_cards; i++) {
        draw_card(env);
      }
      env->remaining_discards--;
    }
  }
  if (env->tick >= 400) {
    env->terminals[0] = 1;
    env->rewards[0] += env->reward_failure_penalty;
    add_log(env);
    c_reset(env);
    return;
  }
  compute_observations(env);
}

// Helper Render Functions
const char *SuitToString(int suit) {
  static const char *names[] = {"Club", "Diamond", "Heart", "Spade"};
  return (suit >= 0 && suit < 4) ? names[suit] : "Unknown";
}

const char *RankToString(int rank) {
  static const char *names[] = {"2", "3", "4", "5", "6", "7", "8",
                                "9", "T", "J", "Q", "K", "A"};
  return (rank >= 0 && rank < 13) ? names[rank] : "?";
}

// Required function. Should handle creating the client on first call
void c_render(Balatro *env) {
  if (!IsWindowReady()) {
    // InitWindow(64*env->size, 64*env->size, "PufferLib Balatro");
    InitWindow(1000, 750, "PufferLib Balatro");
    SetTargetFPS(1);
  }

  // Standard across our envs so exiting is always the same
  if (IsKeyDown(KEY_ESCAPE)) {
    exit(0);
  }

  BeginDrawing();
  ClearBackground((Color){6, 24, 24, 255});

  // TODO: Implement Cards

  int card_width = GetScreenWidth() / env->hand_size;
  int card_height = card_width * 3 / 2;
  int padding = 5;

  for (uint8_t i = 0; i < env->num_cards_in_hand; i++) {
    uint8_t card_idx = env->hand[i];
    Card card = env->cards[card_idx];
    Color card_color = env->selected_mask[card_idx] ? GRAY : RAYWHITE;

    DrawRectangle(
        card_width * i + padding, GetScreenHeight() / 2 - card_height + padding,
        card_width - 2 * padding, card_height - 2 * padding, card_color);
    // DrawText(TextFormat("Hand %d: %s of %ss, Enhancement %d, Seal %d, Edition
    // %d", i, RankToString(card->rank), SuitToString(card->suit),
    // card->enhancement, card->seal, card->edition), 10, 10 + 20 * i, 10,
    // BLACK);
    Color text_color;
    switch (card.suit) {
    case SUIT_CLUBS:
      text_color = SKYBLUE;
      break;
    case SUIT_DIAMONDS:
      text_color = ORANGE;
      break;
    case SUIT_HEARTS:
      text_color = RED;
      break;
    case SUIT_SPADES:
      text_color = BLACK;
      break;
    default:
      text_color = MAGENTA;
      break;
    }

    DrawText(
        TextFormat("%s %s", RankToString(card.rank), SuitToString(card.suit)),
        card_width * i + padding * 2,
        GetScreenHeight() / 2 - card_height + padding, 20, text_color);
  }

  DrawText(TextFormat("Remaining Hands: %d, Remaining Discards: %d",
                      env->remaining_hands, env->remaining_discards),
           10, 10, 10, RAYWHITE);
  DrawText(TextFormat("%d action", env->actions[0]), 10, 25, 10, PURPLE);
  DrawText(TextFormat("%.0f score", env->round_score), 100, 25, 10, PINK);
  DrawText(TextFormat("%d selected cards", env->num_selected_cards), 200, 25,
           10, ORANGE);
  DrawText(TextFormat("%d ticks", env->tick), 300, 25, 10, YELLOW);
  DrawText(TextFormat("%.4f rewards", env->rewards[0]), 400, 25, 10, YELLOW);

  float current_selected_score = calculate_hand_score(env);
  DrawText(TextFormat("%.0f score", current_selected_score), 500, 25, 10,
           RAYWHITE);

  const char *hand_type_string;

  HandTypes current_selected_hand_type = get_hand_type(env);

  switch (current_selected_hand_type) {
  case HAND_FLUSH_FIVE:
    hand_type_string = "Flush Five";
    break;
  case HAND_FLUSH_HOUSE:
    hand_type_string = "Flush House";
    break;
  case HAND_FIVE_OF_A_KIND:
    hand_type_string = "5 of a Kind";
    break;
  case HAND_STRAIGHT_FLUSH:
    hand_type_string = "Straight Flush";
    break;
  case HAND_FOUR_OF_A_KIND:
    hand_type_string = "4 of a Kind";
    break;
  case HAND_FULL_HOUSE:
    hand_type_string = "Full House";
    break;
  case HAND_FLUSH:
    hand_type_string = "Flush";
    break;
  case HAND_STRAIGHT:
    hand_type_string = "Straight";
    break;
  case HAND_THREE_OF_A_KIND:
    hand_type_string = "3 of a Kind";
    break;
  case HAND_TWO_PAIR:
    hand_type_string = "2 Pair";
    break;
  case HAND_PAIR:
    hand_type_string = "Pair";
    break;
  case HAND_HIGH_CARD:
    hand_type_string = "High Card";
    break;
  case -1:
    hand_type_string = "None";
    break;
  case -2:
    hand_type_string = "Error";
    break;
  default:
    hand_type_string = "Unknown";
    break;
  }
  DrawText(TextFormat("%s", hand_type_string), 600, 25, 10, RAYWHITE);

  DrawText(TextFormat("Current Run Seed: %u", env->run_seed), 700, 25, 10,
           GREEN);

  // int px = 64;
  // for (int i = 0; i < env->size; i++) {
  //     for (int j = 0; j < env->size; j++) {
  //         int tex = env->observations[i*env->size + j];
  //         if (tex == EMPTY) {
  //             continue;
  //         }
  //         Color color = (tex == AGENT) ? (Color){0, 187, 187, 255} :
  //         (Color){187, 0, 0, 255}; DrawRectangle(j*px, i*px, px, px, color);
  //     }
  // }

  EndDrawing();
}

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(Balatro *env) {
  if (IsWindowReady()) {
    CloseWindow();
  }
}
