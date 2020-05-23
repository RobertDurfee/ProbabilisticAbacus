def get_chips(transition_matrix, critical_chips, transient_states):
    chips = []
    for state in range(len(transition_matrix)):
        if state in transient_states:
            chips.append(critical_chips[state])
        else:
            chips.append(0)
    chips[0] += 1
    while True:
        any_move = False
        for transient_state in transient_states:
            while chips[transient_state] > critical_chips[transient_state]:
                for next_state, transition_chips in enumerate(transition_matrix[transient_state]):
                    if transition_chips > 0:
                        any_move = True
                        chips[transient_state] -= transition_chips
                        chips[next_state] += transition_chips
        if not any_move:
            chips[0] = critical_chips[0]
            all_critical = True
            for transient_state in transient_states:
                if chips[transient_state] != critical_chips[transient_state]:
                    all_critical = False
            if all_critical:
                return chips
            else:
                chips[0] += 1

def get_absorption_probabilities(transition_matrix):
    critical_chips = []
    for transition_row in transition_matrix:
        critical_chips.append(sum(transition_row) - 1)
    transient_states = []
    terminal_states = []
    for state in range(len(transition_matrix)):
        if critical_chips[state] >= 0:
            transient_states.append(state)
        elif critical_chips[state] == -1:
            terminal_states.append(state)
    chips = get_chips(transition_matrix, critical_chips, transient_states)
    total_chips = 0
    for terminal_state in terminal_states:
        total_chips += chips[terminal_state]
    absorption_probabilities = []
    for terminal_state in terminal_states:
        absorption_probabilities.append(float(chips[terminal_state]) / total_chips)
    return absorption_probabilities

