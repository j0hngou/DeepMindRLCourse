import numpy as np
import itertools
import matplotlib.pyplot as plt


class Card:

    def __init__(self, card):
        """
        card: string of length 2
        i.e. A♦, 2♣, etc.
        """
        self.card = card
        self.value = self.get_value()
        self.suit = self.get_suit()
        self.is_ace = self.is_ace()

    def get_value(self):
        """
        Returns the value of the card.
        """
        if self.card[0] == 'A':
            return 1
        elif self.card[0] == 'J' or self.card[0] == 'Q' or self.card[
                0] == 'K' or self.card[0] == '1':
            return 10
        else:
            return int(self.card[0])

    def get_suit(self):
        """
        Returns the suit of the card.
        """
        return self.card[1]

    def is_ace(self):
        """
        Returns True if the card is an ace.
        """
        return self.card[0] == 'A'

    def __str__(self):
        return self.card

    def __repr__(self):
        return str(self.card)


class Deck:

    def __init__(self):
        faces = [
            'A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'
        ]  # face cards
        suits = ['♣', '♦', '♥', '♠']
        self.cards = [Card(face + suit) for face in faces for suit in suits]
        self.shuffle()
        self.cards_left = len(self.cards)
        self.dealt_cards = []
        self.dealt_cards_left = 0
        self.dealt_cards_total = 0

    def shuffle(self):
        np.random.shuffle(self.cards)

    def deal_card(self):
        self.dealt_cards_left += 1
        self.dealt_cards_total += 1
        self.cards_left -= 1
        return self.cards[-self.cards_left]

    def deal_hand(self, hand_size):
        self.dealt_cards_left += hand_size
        self.dealt_cards_total += hand_size
        self.cards_left -= hand_size
        return self.cards[-self.cards_left:-self.cards_left + hand_size]


class Hand:

    def __init__(self):
        self.cards = []
        self.value = 0
        self.used_ace = False

    def add_card(self, card):
        self.cards.append(card)
        self.value = self.calculate_value()

    def has_ace(self):
        if self.used_ace == True:
            return False
        else:
            return any(card.is_ace for card in self.cards)

    def calculate_value(self):
        if self.has_ace():
            value = sum([card.value for card in self.cards if card.value != 1])
            if value + 10 <= 21:
                return value + 10
            else:
                self.used_ace = True
                return value + 1
        else:
            return sum([card.value for card in self.cards])

    def is_bust(self):
        return self.value > 21

    def __str__(self):
        return str(self.cards)

    def __repr__(self):
        return str(self.cards)


class BlackJack:

    def __init__(self, player_strategy=None):
        self.deck = Deck()
        self.episodes = 0
        self.hands = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.players = [0, 1]
        self.player_strategy = player_strategy if player_strategy else self.dealer_strategy_soft17
        self.dealer_strategy = self.dealer_strategy_soft17

    def dealer_strategy_soft17(self, hand: Hand):
        has_ace = hand.has_ace()
        if hand.value == 17 and has_ace:
            return 'H'
        elif hand.value == 17 and not has_ace:
            return 'S'
        elif hand.value < 17:
            return 'H'
        else:
            return 'S'

    def deal_hand(self):
        hand = Hand()
        hand.add_card(self.deal_card())
        hand.add_card(self.deal_card())
        return hand

    def deal_card(self):
        return self.deck.deal_card()

    def new_deck(self):
        self.deck = Deck()
        self.episodes += 1

    def play_episode(self):
        self.new_deck()
        player1hand = Hand()
        player2hand = Hand()
        player1hand.add_card(self.deal_card())
        player1hand.add_card(self.deal_card())
        player2hand.add_card(self.deal_card())
        player2hand.add_card(self.deal_card())
        player1strategy = self.player_strategy(player1hand)
        player2strategy = self.player_strategy(player2hand)
        while player1strategy == 'H':
            player1hand.add_card(self.deal_card())
            player1strategy = self.player_strategy(player1hand)
        while player2strategy == 'H':
            player2hand.add_card(self.deal_card())
            player2strategy = self.player_strategy(player2hand)
        if player1hand.value > 21:
            self.losses += 1
            reward = -1
        elif player2hand.value > 21:
            self.wins += 1
            reward = 1
        elif player1hand.value > player2hand.value:
            self.wins += 1
            reward = 1
        elif player1hand.value < player2hand.value:
            self.losses += 1
            reward = -1
        else:
            self.draws += 1
            reward = 0
        return reward

    def play_n_episodes(self, n: int):
        for _ in range(n):
            self.play_episode()
        return self.wins / n, self.losses / n, self.draws / n


class Agent:

    def __init__(self,
                 blackjack: BlackJack,
                 epsilon: float,
                 alpha: float,
                 gamma: float,
                 policy=None):
        self.blackjack = blackjack
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.Q = self.init_Q()
        self.N_Q = self.init_N_Q()
        self.V = self.init_V()
        self.N_V = self.init_N_V()
        self.episodes = 0
        self.hands = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.policy = policy if policy else self.dealer_strategy_soft17
        self.dealer_strategy = self.dealer_strategy_soft17

    def dealer_strategy_soft17(self, hand: Hand):
        has_ace = hand.has_ace()
        if hand.value == 17 and has_ace:
            return 'H'
        elif hand.value == 17 and not has_ace:
            return 'S'
        elif hand.value < 17:
            return 'H'
        else:
            return 'S'

    def player_strategy_soft20(self, hand: Hand):
        has_ace = hand.has_ace()
        if hand.value == 20 and has_ace:
            return 'H'
        elif hand.value == 20 and not has_ace:
            return 'S'
        elif hand.value < 20:
            return 'H'
        else:
            return 'S'

    def possible_states(self):
        playerhandvalue = np.arange(12, 23)
        dealerhandvalue = np.arange(2, 11)
        useable_ace = [True, False]
        possible_states = itertools.product(playerhandvalue, dealerhandvalue,
                                            useable_ace)
        return possible_states

    def init_Q(self):
        Q = {}
        possible_states = self.possible_states()
        for state in possible_states:
            Q[state] = {}
            for action in ['H', 'S']:
                Q[state][action] = 0
        return Q

    def init_N_Q(self):
        N = {}
        possible_states = self.possible_states()
        for state in possible_states:
            N[state] = {}
            for action in ['H', 'S']:
                N[state][action] = 0
        return N

    def init_V(self):
        V = {}
        possible_states = self.possible_states()
        for state in possible_states:
            V[state] = 0
        return V

    def init_N_V(self):
        N = {}
        possible_states = self.possible_states()
        for state in possible_states:
            N[state] = 0
        return N

    def temporal_difference_V(self, states: tuple[int, int, bool],
                              reward: int):
        """
        Implements the TD(0) update rule for the value function.
        """
        G = self.gamma
        oldstate_estimates = []
        for state in states:
            oldstate_estimates.append(self.V[state])
        for k, state in enumerate(reversed(states)):
            self.N_V[state] = self.N_V[state] + 1
            if k == 0:
                self.V[state] = self.V[state] + self.alpha * (reward -
                                                              self.V[state])
            else:
                self.V[state] = self.V[state] + self.alpha * (
                    self.gamma**k * oldstate_estimates[-k] -
                    self.V[state]) / self.N_V[state]

    def monte_carlo_V(self, states: list[tuple[int, int, bool]], reward: int):
        G = reward
        for k, state in enumerate(states[::-1], 0):
            self.N_V[state] = self.N_V[state] + 1
            G = self.gamma**k * G
            self.V[state] = self.V[state] + (G -
                                             self.V[state]) / self.N_V[state]

    def value_V(self, state: tuple[int, int, bool]):
        return self.V[state]

    def value_Q(self, state: tuple[int, int, bool]):
        return max(self.Q[state]['H'], self.Q[state]['S'])

    def play_episode(self, learning_strategy: str):
        state: list(int, int) = [0, 0, False]
        states = []
        self.blackjack.new_deck()
        player1hand = Hand()
        player2hand = Hand()
        player1hand.add_card(self.blackjack.deal_card())
        player1hand.add_card(self.blackjack.deal_card())
        state[0] = player1hand.value
        state[2] = player1hand.has_ace()
        player2hand.add_card(self.blackjack.deal_card())
        state[1] = player2hand.value
        if player1hand.value >= 12:
            states.append(tuple(state))
        player2hand.add_card(self.blackjack.deal_card())
        player1strategy = self.policy(player1hand)
        player2strategy = self.dealer_strategy(player2hand)
        while player1strategy == 'H':
            player1hand.add_card(self.blackjack.deal_card())
            player1strategy = self.policy(player1hand)
            state[0] = player1hand.value if not player1hand.is_bust() else 22
            state[2] = player1hand.has_ace()
            if player1hand.value >= 12:
                states.append(tuple(state))
        while player2strategy == 'H':
            player2hand.add_card(self.blackjack.deal_card())
            player2strategy = self.dealer_strategy(player2hand)
        if player1hand.value > 21:
            self.losses += 1
            reward = -1
        elif player2hand.value > 21:
            self.wins += 1
            reward = 1
        elif player1hand.value > player2hand.value:
            self.wins += 1
            reward = 1
        elif player1hand.value < player2hand.value:
            self.losses += 1
            reward = -1
        else:
            self.draws += 1
            reward = 0

        if learning_strategy == 'temporal_difference_V':
            self.temporal_difference_V(states, reward)
        elif learning_strategy == 'monte_carlo_V':
            self.monte_carlo_V(states, reward)
        elif learning_strategy == 'temporal_Q':
            raise NotImplementedError()
        elif learning_strategy == 'monte_carlo_Q':
            self.monte_carlo_Q(states, player1strategy, reward)
        self.episodes += 1
        self.hands += 1

    def play_n_episodes(self, n: int, learning_strategy: str):
        for i in range(n):
            self.play_episode(learning_strategy)

    # Following 2 functions are adaptations based on https://github.com/mtrazzi/rl-book-challenge/blob/master/chapter5/figures.py
    def values_to_grid(self, has_ace_plot: bool = False):
        playerhandvalue = np.arange(12, 22)
        dealerhandvalue = np.arange(2, 11)
        if has_ace_plot:
            states = itertools.product(playerhandvalue, dealerhandvalue,
                                       [True])
        else:
            states = itertools.product(playerhandvalue, dealerhandvalue,
                                       [False])
        values = np.zeros((len(playerhandvalue), len(dealerhandvalue)))
        for (i, (player_sum, dealer_card, usab)) in enumerate(states):
            values[player_sum - 12, dealer_card - 2] = self.value_V(
                (player_sum, dealer_card, usab))
        return values

    def print_plot(self, to_print, title, fig_id):
        """Prints the grid `to_print` as presented in Figure 5.1. and 5.3."""
        dealer_idxs = np.arange(2, 11)
        player_idxs = np.arange(12, 22)
        fig = plt.figure()
        ax = fig.add_subplot(fig_id, projection='3d')
        ax.set_title(title, fontsize=10)
        (X, Y), Z = np.meshgrid(dealer_idxs, player_idxs), to_print
        ax.set_xlabel('Dealer showing', fontsize=8)
        ax.set_ylabel('Player Sum', fontsize=8)
        ax.set_xticks([dealer_idxs.min(), dealer_idxs.max()])
        ax.set_yticks([player_idxs.min(), player_idxs.max()])
        ax.set_zticks([-1, 1])
        ax.plot_surface(X, Y, Z)
        plt.show()


if __name__ == '__main__':
    blackjack = BlackJack()
    agent = Agent(blackjack, epsilon=0.1, alpha=0.03, gamma=1)
    agent.play_n_episodes(500000, 'temporal_difference_V')
    print(f'Wins: {agent.wins}')
    print(f'Losses: {agent.losses}')
    print(f'Draws: {agent.draws}')
    agent.print_plot(agent.values_to_grid(has_ace_plot=True), 'Value Function',
                     221)
