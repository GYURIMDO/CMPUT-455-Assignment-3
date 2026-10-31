"""
Private tests from claude
"""
# Run with: python3 extra_tests_a3.py

import contextlib
import io
import random
import sys
import time

sys.dont_write_bytecode = True 

from a3 import COLORS, CommandInterface

TIME_LIMIT = 1.0  # seconds per command, as in a3test.py
SAMPLES = 20000
TOLERANCE = 0.02

passed = 0
failed = 0


def check(description: str, condition: bool) -> None:
    global passed, failed
    if condition:
        passed += 1
        print("PASS  " + description)
    else:
        failed += 1
        print("FAIL  " + description)


def new_interface(komi: str, heaps: str, policy: str = "random") -> CommandInterface:
    interface = CommandInterface()
    interface.cmd_heapgo(komi + " " + heaps)
    interface.cmd_policy(policy)
    return interface


def snapshot(interface: CommandInterface):
    game = interface.game
    return (str(game), dict(game.score), game.toplay)


def run_command(command, args: str):
    """Run a command, returning (success, stdout text)."""
    output = io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
        success = command(args)
    return success, output.getvalue().strip()


def large_game_text() -> str:
    heaps = []
    for heap_number in range(20):
        heap = []
        for position in range(20):
            if (heap_number + position) % 2:
                heap.append(('b', 1 + (heap_number * position) % 20))
            else:
                heap.append(('w', 1 + (heap_number + position) % 20))
        heaps.append(heap)
    return str(heaps)


def test_simulate_preserves_state() -> None:
    for policy in ("random", "probabilistic"):
        interface = new_interface("0.5", "[[('b',5),('w',3)],[('w',9)],[('b',2)]]", policy)
        before = snapshot(interface)
        winners = set()
        for _ in range(50):
            success, winner = run_command(interface.cmd_simulate, "")
            winners.add(winner)
        check("simulate returns b or w and succeeds ({})".format(policy),
              success and winners <= {"b", "w"})
        check("simulate leaves the game unchanged ({})".format(policy),
              snapshot(interface) == before)


def test_genmove_plays_exactly_one_move() -> None:
    for policy in ("random", "probabilistic"):
        interface = new_interface("0.5", "[[('b',5),('w',3)],[('w',9)],[('b',2)]]", policy)
        expected = interface.game.copy()
        success, output = run_command(interface.cmd_genmove, "")
        move = int(output)
        expected.play(move)
        check("genmove plays exactly the move it prints ({})".format(policy),
              success and str(interface.game) == str(expected)
              and interface.game.toplay == expected.toplay
              and interface.game.score == expected.score)


def test_genmove_finds_clear_winner() -> None:
    # Black wins by taking heap 0 (20 points); taking heap 1 first loses.
    for policy in ("random", "probabilistic"):
        picks = []
        for _ in range(20):
            interface = new_interface("0.5", "[[('w',20)],[('b',1)]]", policy)
            interface.cmd_numsimulations("20")
            _, output = run_command(interface.cmd_genmove, "")
            picks.append(int(output))
        check("genmove picks the clearly winning move ({})".format(policy),
              all(move == 0 for move in picks))


def test_numsimulations_arguments() -> None:
    interface = new_interface("0.5", "[[('b',5)]]")
    check("default number of simulations is 10", interface.num_simulations == 10)
    for valid in ("1", "100", "37"):
        success, _ = run_command(interface.cmd_numsimulations, valid)
        check("numsimulations {} accepted".format(valid),
              success and interface.num_simulations == int(valid))
    interface.cmd_numsimulations("50")
    for invalid in ("0", "101", "-3", "abc", "", "5 6"):
        success, _ = run_command(interface.cmd_numsimulations, invalid)
        check("numsimulations '{}' rejected, old value kept".format(invalid),
              (not success) and interface.num_simulations == 50)


def test_policymoves_output() -> None:
    # heaps cannot start empty, so empty heap 2 by playing it, then give black the move back
    interface = new_interface("0.5", "[[('b',5),('w',3)],[('w',9)],[('w',4)],[('b',2)]]", "random")
    interface.game.play(2)
    interface.cmd_toplay("b")
    _, output = run_command(interface.cmd_policymoves, "")
    check("policymoves random skips empty heaps", output == "0 0.333 1 0.333 3 0.333")
    interface.cmd_policy("probabilistic")
    _, output = run_command(interface.cmd_policymoves, "")
    check("policymoves probabilistic uses move weights", output == "0 0.214 1 0.643 3 0.143")
    interface = new_interface("0.5", "[[('b',1)]]")
    interface.game.play(0)
    success, output = run_command(interface.cmd_policymoves, "")
    check("policymoves at the end of the game prints nothing", success and output == "")


def sample_playout_move(interface: CommandInterface) -> int:
    """One move chosen by the policy the playouts use, from the current game."""
    game = interface.game
    interface.solver.prepare(game)
    sizes = [len(heap) for heap in game.heaps]
    gain = interface.solver.gain[COLORS.index(game.toplay)]
    return interface.choose_playout_move(game.legal_moves(), sizes, gain)


def test_policy_frequencies() -> None:
    interface = new_interface("0.5", "[[('b',5),('w',3)],[('w',9)],[('b',2)]]", "probabilistic")
    expected = interface.move_probabilities(interface.game)
    counts = {move: 0 for move in expected}
    for _ in range(SAMPLES):
        counts[sample_playout_move(interface)] += 1
    check("probabilistic playout moves follow the weights",
          all(abs(counts[move] / SAMPLES - expected[move]) < TOLERANCE for move in expected))
    interface.cmd_policy("random")
    counts = {move: 0 for move in expected}
    for _ in range(SAMPLES):
        counts[sample_playout_move(interface)] += 1
    check("random playout moves are uniform",
          all(abs(counts[move] / SAMPLES - 1 / 3) < TOLERANCE for move in expected))


def test_simulation_respects_policy() -> None:
    # Heap 0 gives black 20 points and wins; heap 1 gives 1 point and loses.
    # With the probabilistic policy black picks heap 0 first 20/21 of the time,
    # with the random policy only half of the time, so black's win rate differs.
    rates = {}
    for policy in ("random", "probabilistic"):
        interface = new_interface("0.5", "[[('w',20)],[('b',1)]]", policy)
        wins = sum(run_command(interface.cmd_simulate, "")[1] == "b" for _ in range(SAMPLES // 4))
        rates[policy] = wins / (SAMPLES // 4)
    check("simulate follows the policy (random about 0.50, probabilistic about 0.95)",
          abs(rates["random"] - 0.5) < 0.05 and abs(rates["probabilistic"] - 20 / 21) < 0.05)


def reference_playout(interface: CommandInterface, game) -> str:
    """Slow playout on the real game rules, to compare the fast one against."""
    simulation = game.copy()
    while not simulation.game_over():
        moves = simulation.legal_moves()
        if interface.policy == "random":
            move = random.choice(moves)
        else:
            weights = [interface.move_weight(simulation, move) for move in moves]
            move = random.choices(moves, weights=weights, k=1)[0]
        simulation.play(move)
    return simulation.winner()


def test_playout_matches_game_rules() -> None:
    # With the same random seed both playouts make the same random calls,
    # so they must pick the same moves and find the same winner.
    generator = random.Random(455)
    for policy in ("random", "probabilistic"):
        matches = 0
        trials = 300
        for trial in range(trials):
            heaps = [[(generator.choice("bw"), generator.randint(1, 20))
                      for _ in range(generator.randint(1, 8))]
                     for _ in range(generator.randint(1, 8))]
            interface = new_interface(str(generator.randint(-5, 5) + 0.5), str(heaps), policy)
            if generator.random() < 0.5:
                interface.cmd_toplay("w")
            interface.solver.prepare(interface.game)
            random.seed(trial)
            fast = interface.run_simulation(interface.game)
            random.seed(trial)
            slow = reference_playout(interface, interface.game)
            matches += fast == slow
        check("fast playout matches the game rules on {} random positions ({})".format(
            trials, policy), matches == trials)


def test_game_over_and_small_games() -> None:
    interface = new_interface("0.5", "[[('b',1)]]")
    interface.game.play(0)
    success, winner = run_command(interface.cmd_simulate, "")
    check("simulate at the end of the game returns the winner", success and winner == "b")
    interface = new_interface("0.5", "[[('w',4)]]")
    success, output = run_command(interface.cmd_genmove, "")
    check("genmove with a single legal move plays it", success and output == "0")


def test_timing() -> None:
    for policy in ("random", "probabilistic"):
        for simulations in ("10", "100"):
            interface = new_interface("0.5", large_game_text(), policy)
            interface.cmd_numsimulations(simulations)
            start = time.time()
            run_command(interface.cmd_genmove, "")
            elapsed = time.time() - start
            check("genmove on 20x20 within {}s: {}, n={} ({:.2f}s)".format(
                TIME_LIMIT, policy, simulations, elapsed), elapsed <= TIME_LIMIT)


if __name__ == "__main__":
    test_simulate_preserves_state()
    test_genmove_plays_exactly_one_move()
    test_genmove_finds_clear_winner()
    test_numsimulations_arguments()
    test_policymoves_output()
    test_policy_frequencies()
    test_simulation_respects_policy()
    test_playout_matches_game_rules()
    test_game_over_and_small_games()
    test_timing()
    print("\n{} passed, {} failed".format(passed, failed))
    sys.exit(1 if failed else 0)
