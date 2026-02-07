from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional, Tuple
from .constants import TILE_NAMES, TILE_COUNT, RED_TILE_ID, TILE_NAME_TO_ID, tile_to_name

def hand_to_counts(hand: List[int]) -> List[int]:
    counts = [0] * TILE_COUNT
    for t in hand:
        if 0 <= t < TILE_COUNT:
            counts[t] += 1
    return counts


class Rule:
    name = ""
    weight = 0.0

    def apply(self, state: Dict) -> List[float]:
        return [0.0] * TILE_COUNT


class QueMenRule(Rule):
    name = "que_men"
    weight = 0.25

    def apply(self, state: Dict) -> List[float]:
        scores = [0.0] * TILE_COUNT
        hand = state.get("hand", [])
        color_counts = [0, 0, 0]
        for tile in hand:
            if 0 <= tile < 27:
                color_counts[tile // 9] += 1
        que_color = min(range(3), key=lambda c: color_counts[c])
        start = que_color * 9
        for tile in range(start, start + 9):
            value = tile % 9
            if 2 <= value <= 6:
                scores[tile] = 1.0
            elif value in (0, 8):
                scores[tile] = 0.5
            else:
                scores[tile] = 0.7
        return scores


class DaziCompletionRule(Rule):
    name = "dazi_completion"
    weight = 0.2

    def apply(self, state: Dict) -> List[float]:
        scores = [0.0] * TILE_COUNT
        hand = state.get("hand", [])
        dazi_list = self._identify_dazi(hand)
        for dazi_type, dazi_tiles in dazi_list:
            if dazi_type == "liang_mian":
                for target in self._get_liangmian_targets(dazi_tiles):
                    scores[target] += 1.0
            elif dazi_type == "bian_zhang":
                target = self._get_bianzhang_target(dazi_tiles)
                scores[target] += 0.8
            elif dazi_type == "kan_zhang":
                target = self._get_kanzhang_target(dazi_tiles)
                scores[target] += 0.7
            elif dazi_type == "dui_zi":
                scores[dazi_tiles[0]] += 0.6
            elif dazi_type == "gu_zhang":
                for neighbor in self._get_possible_neighbors(dazi_tiles[0]):
                    scores[neighbor] += 0.3
        return scores

    def _identify_dazi(self, hand: List[int]) -> List[Tuple[str, List[int]]]:
        dazi_list: List[Tuple[str, List[int]]] = []
        counts = hand_to_counts(hand)
        for tile, count in enumerate(counts):
            if count >= 2:
                dazi_list.append(("dui_zi", [tile, tile]))
        for base in (0, 9, 18):
            for v in range(9):
                t = base + v
                if counts[t] == 0:
                    continue
                if v <= 7 and counts[t + 1] > 0:
                    if v == 0 or v + 1 == 8:
                        dazi_list.append(("bian_zhang", [t, t + 1]))
                    else:
                        dazi_list.append(("liang_mian", [t, t + 1]))
                if v <= 6 and counts[t + 2] > 0:
                    dazi_list.append(("kan_zhang", [t, t + 2]))
        for tile, count in enumerate(counts[:27]):
            if count == 1:
                dazi_list.append(("gu_zhang", [tile]))
        return dazi_list

    def _get_liangmian_targets(self, tiles: List[int]) -> List[int]:
        t1, t2 = sorted(tiles)
        targets = []
        if t1 % 9 > 0:
            targets.append(t1 - 1)
        if t2 % 9 < 8:
            targets.append(t2 + 1)
        return targets

    def _get_bianzhang_target(self, tiles: List[int]) -> int:
        t1, t2 = sorted(tiles)
        if t1 % 9 == 0:
            return t2 + 1
        return t1 - 1

    def _get_kanzhang_target(self, tiles: List[int]) -> int:
        t1, t2 = sorted(tiles)
        return (t1 + t2) // 2

    def _get_possible_neighbors(self, tile: int) -> List[int]:
        neighbors = []
        if tile % 9 > 0:
            neighbors.append(tile - 1)
        if tile % 9 < 8:
            neighbors.append(tile + 1)
        return neighbors


class ShantenCalculator:
    def __init__(self) -> None:
        self.cache: Dict[Tuple[int, ...], int] = {}

    def calculate(self, hand: List[int]) -> int:
        counts = hand_to_counts(hand)
        key = tuple(counts)
        if key in self.cache:
            return self.cache[key]
        value = self._standard_shanten(counts)
        self.cache[key] = value
        return value

    def _standard_shanten(self, counts: List[int]) -> int:
        pairs = [i for i, c in enumerate(counts) if c >= 2]
        candidates = [None] + pairs
        min_shanten = 8
        for pair_tile in candidates:
            c = counts.copy()
            pair_count = 0
            if pair_tile is not None:
                c[pair_tile] -= 2
                pair_count = 1
            melds, taatsu = self._max_melds_taatsu(c)
            taatsu = min(taatsu, 4 - melds)
            shanten = 8 - melds * 2 - taatsu - pair_count
            min_shanten = min(min_shanten, shanten)
        return min_shanten

    def _max_melds_taatsu(self, counts: List[int]) -> Tuple[int, int]:
        honor_melds = 0
        honor_taatsu = 0
        honor_melds += counts[RED_TILE_ID] // 3
        if counts[RED_TILE_ID] % 3 == 2:
            honor_taatsu += 1
        suits = []
        for base in (0, 9, 18):
            suit_counts = counts[base : base + 9]
            suits.append(self._analyze_suit(tuple(suit_counts)))
        combos = {(0, 0)}
        for suit_result in suits:
            new_combos = set()
            for m1, t1 in combos:
                for m2, t2 in suit_result:
                    new_combos.add((m1 + m2, t1 + t2))
            combos = new_combos
        max_m = 0
        max_t = 0
        for m, t in combos:
            if m > max_m or (m == max_m and t > max_t):
                max_m = m
                max_t = t
        return max_m + honor_melds, max_t + honor_taatsu

    def _analyze_suit(self, counts: Tuple[int, ...]) -> List[Tuple[int, int]]:
        memo: Dict[Tuple[int, ...], List[Tuple[int, int]]] = {}

        def dfs(state: Tuple[int, ...]) -> List[Tuple[int, int]]:
            if state in memo:
                return memo[state]
            counts_list = list(state)
            i = 0
            while i < 9 and counts_list[i] == 0:
                i += 1
            if i >= 9:
                memo[state] = [(0, 0)]
                return memo[state]
            results: List[Tuple[int, int]] = []
            if counts_list[i] >= 3:
                counts_list[i] -= 3
                for m, t in dfs(tuple(counts_list)):
                    results.append((m + 1, t))
                counts_list[i] += 3
            if i <= 6 and counts_list[i] > 0 and counts_list[i + 1] > 0 and counts_list[i + 2] > 0:
                counts_list[i] -= 1
                counts_list[i + 1] -= 1
                counts_list[i + 2] -= 1
                for m, t in dfs(tuple(counts_list)):
                    results.append((m + 1, t))
                counts_list[i] += 1
                counts_list[i + 1] += 1
                counts_list[i + 2] += 1
            if counts_list[i] >= 2:
                counts_list[i] -= 2
                for m, t in dfs(tuple(counts_list)):
                    results.append((m, t + 1))
                counts_list[i] += 2
            if i <= 7 and counts_list[i] > 0 and counts_list[i + 1] > 0:
                counts_list[i] -= 1
                counts_list[i + 1] -= 1
                for m, t in dfs(tuple(counts_list)):
                    results.append((m, t + 1))
                counts_list[i] += 1
                counts_list[i + 1] += 1
            if i <= 6 and counts_list[i] > 0 and counts_list[i + 2] > 0:
                counts_list[i] -= 1
                counts_list[i + 2] -= 1
                for m, t in dfs(tuple(counts_list)):
                    results.append((m, t + 1))
                counts_list[i] += 1
                counts_list[i + 2] += 1
            counts_list[i] -= 1
            for m, t in dfs(tuple(counts_list)):
                results.append((m, t))
            counts_list[i] += 1
            memo[state] = results
            return results

        return dfs(counts)


class ShantenReductionRule(Rule):
    name = "shanten_reduction"
    weight = 0.18

    def __init__(self, calculator: ShantenCalculator) -> None:
        self.calculator = calculator

    def apply(self, state: Dict) -> List[float]:
        scores = [0.0] * TILE_COUNT
        hand = state.get("hand", [])
        current_shanten = self.calculator.calculate(hand)
        for candidate in range(TILE_COUNT):
            test_hand = hand + [candidate]
            new_shanten = self.calculator.calculate(test_hand)
            shanten_reduction = current_shanten - new_shanten
            if shanten_reduction > 0:
                scores[candidate] = shanten_reduction * 0.5
                if new_shanten <= 0:
                    scores[candidate] += 0.3
        return scores


class DangerAvoidanceRule(Rule):
    name = "danger_avoidance"
    weight = 0.15

    def apply(self, state: Dict) -> List[float]:
        scores = [0.0] * TILE_COUNT
        discarded = state.get("discarded_tiles", [])
        current_player = state.get("current_player", 0)
        player_discards: Dict[int, List[int]] = {i: [] for i in range(4)}
        for discard in discarded:
            player_discards.setdefault(discard["player"], []).append(discard["tile"])
        dangerous_tiles = self._identify_dangerous_tiles(player_discards, current_player)
        for tile in dangerous_tiles:
            if 0 <= tile < TILE_COUNT:
                scores[tile] -= 0.8
        safe_tiles = self._identify_safe_tiles(player_discards)
        for tile in safe_tiles:
            scores[tile] += 0.4
        return scores

    def _identify_dangerous_tiles(self, player_discards: Dict[int, List[int]], current_player: int) -> List[int]:
        dangerous = set()
        for player_id, discards in player_discards.items():
            if player_id == current_player:
                continue
            if not discards:
                continue
            recent_discards = discards[-5:]
            for tile in recent_discards:
                if 0 <= tile < 27:
                    color = tile // 9
                    value = tile % 9
                    if value > 0:
                        dangerous.add(color * 9 + (value - 1))
                    if value < 8:
                        dangerous.add(color * 9 + (value + 1))
        return list(dangerous)

    def _identify_safe_tiles(self, player_discards: Dict[int, List[int]]) -> List[int]:
        safe = set()
        all_discarded = []
        for discards in player_discards.values():
            all_discarded.extend(discards)
        counts = hand_to_counts(all_discarded)
        for tile, count in enumerate(counts):
            if count >= 2:
                safe.add(tile)
        return list(safe)


class TileEfficiencyRule(Rule):
    name = "tile_efficiency"
    weight = 0.12

    def __init__(self, calculator: ShantenCalculator) -> None:
        self.calculator = calculator

    def apply(self, state: Dict) -> List[float]:
        scores = [0.0] * TILE_COUNT
        hand = state.get("hand", [])
        for candidate in range(TILE_COUNT):
            test_hand = hand + [candidate]
            effective_tiles = self._count_effective_tiles(test_hand)
            scores[candidate] = min(effective_tiles / 10, 1.0)
        return scores

    def _count_effective_tiles(self, hand: List[int]) -> int:
        effective_count = 0
        current_shanten = self.calculator.calculate(hand)
        for test_tile in range(TILE_COUNT):
            test_hand = hand + [test_tile]
            new_shanten = self.calculator.calculate(test_hand)
            if new_shanten < current_shanten:
                effective_count += 1
        return effective_count


class GamePhaseRule(Rule):
    name = "game_phase"
    weight = 0.1

    def apply(self, state: Dict) -> List[float]:
        scores = [0.0] * TILE_COUNT
        round_num = state.get("round", 1)
        if round_num <= 3:
            for tile in range(27):
                value = tile % 9
                if 2 <= value <= 6:
                    scores[tile] += 0.3
        return scores


class PlayerStyleRule(Rule):
    name = "player_style"
    weight = 0.08

    def apply(self, state: Dict) -> List[float]:
        scores = [0.0] * TILE_COUNT
        players = state.get("players", [])
        current_player = state.get("current_player", 0)
        if 0 <= current_player < len(players):
            style = players[current_player].get("style", "balanced")
            if style == "aggressive":
                for tile in range(27):
                    scores[tile] += 0.1
        return scores


class PatternRecognitionRule(Rule):
    name = "pattern_recognition"
    weight = 0.07

    def apply(self, state: Dict) -> List[float]:
        scores = [0.0] * TILE_COUNT
        hand = state.get("hand", [])
        counts = hand_to_counts(hand)
        suit_counts = [sum(counts[i * 9 : i * 9 + 9]) for i in range(3)]
        max_suit = max(range(3), key=lambda i: suit_counts[i])
        if suit_counts[max_suit] >= 9:
            start = max_suit * 9
            for tile in range(start, start + 9):
                scores[tile] += 0.3
        return scores


class RuleBasedRedZhongPredictor:
    def __init__(self) -> None:
        self.calculator = ShantenCalculator()
        self.rules: List[Rule] = [
            QueMenRule(),
            DaziCompletionRule(),
            ShantenReductionRule(self.calculator),
            DangerAvoidanceRule(),
            TileEfficiencyRule(self.calculator),
            GamePhaseRule(),
            PlayerStyleRule(),
            PatternRecognitionRule(),
        ]
        self.rule_weights = self._calculate_rule_weights()

    def _calculate_rule_weights(self) -> Dict[str, float]:
        weights = {rule.name: rule.weight for rule in self.rules}
        total = sum(weights.values()) or 1.0
        for key in list(weights.keys()):
            weights[key] /= total
        return weights

    def predict(self, game_state: Dict) -> List[float]:
        scores = [0.0] * TILE_COUNT
        for rule in self.rules:
            rule_scores = rule.apply(game_state)
            weight = self.rule_weights.get(rule.name, 0.0)
            for tile in range(TILE_COUNT):
                scores[tile] += rule_scores[tile] * weight
        return self._normalize_scores(scores)

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        min_value = min(scores)
        if min_value < 0:
            scores = [s - min_value for s in scores]
        total = sum(scores)
        if total == 0:
            return scores
        return [s / total for s in scores]


class ContextAnalyzer:
    def analyze(self, game_state: Dict) -> Dict:
        round_num = game_state.get("round", 1)
        scores = game_state.get("scores", [])
        current_player = game_state.get("current_player", 0)
        factors = {
            "early_game": 1.0 if round_num <= 3 else 0.0,
            "late_game": 1.0 if round_num >= 8 else 0.0,
            "ahead": 0.0,
            "behind": 0.0,
            "defensive": 0.0,
            "offensive": 0.0,
        }
        if scores and 0 <= current_player < len(scores):
            current_score = scores[current_player]
            avg_score = sum(scores) / len(scores)
            if current_score > avg_score:
                factors["ahead"] = 1.0
                factors["defensive"] = 0.6
            else:
                factors["behind"] = 1.0
                factors["offensive"] = 0.7
        return {"factors": factors}


class LuzhouScorer:
    def __init__(self, calculator: ShantenCalculator) -> None:
        self.calculator = calculator

    def score(self, hand: List[int], game_state: Dict) -> Dict:
        hand = hand[:14]
        counts = hand_to_counts(hand)
        require_que_men = bool(game_state.get("require_que_men", False))
        self_draw = bool(game_state.get("self_draw", True))
        gang_shang_hua = bool(game_state.get("gang_shang_hua", False))
        gang_shang_pao = bool(game_state.get("gang_shang_pao", False))
        tian_hu = bool(game_state.get("tian_hu", False))
        di_hu = bool(game_state.get("di_hu", False))
        fan_cap = int(game_state.get("fan_cap", 40))
        red_in_hand = counts[RED_TILE_ID]
        ghost_total = red_in_hand
        if require_que_men and self._suits_used(counts) >= 3:
            return {"fan": 0, "label": "缺一门不满足", "melds": []}
        qidui = self._qidui_decomposition(counts, ghost_total)
        standard = self._standard_decomposition(counts, ghost_total)
        best = None
        for candidate in (qidui, standard):
            if candidate is None:
                continue
            fan, label = self._score_from_decomposition(candidate, counts, game_state)
            if best is None or fan > best[0]:
                best = (fan, label, candidate["melds"])
        if best is None:
            return {"fan": 0, "label": "未成胡", "melds": []}
        fan, label, melds = best
        if tian_hu or di_hu:
            fan = max(fan, 40)
            label = "天胡" if tian_hu else "地胡"
        if gang_shang_hua:
            fan = 10 if label == "平胡" else fan * 2
            label = f"{label} 杠上花"
        if gang_shang_pao:
            fan = 5 if label == "平胡" else fan * 2
            label = f"{label} 杠上炮"
        if red_in_hand == 0:
            fan *= 2
            label = f"{label} 无鬼"
        if not self_draw:
            fan = max(1, fan // 2)
            label = f"{label} 点炮"
        fan = min(fan, fan_cap)
        return {"fan": fan, "label": label, "melds": melds}

    def _score_from_decomposition(self, decomposition: Dict, counts: List[int], game_state: Dict) -> Tuple[int, str]:
        base = decomposition["type"]
        gui_count = sum(1 for c in counts[:27] if c >= 4)
        is_qing = self._is_qingyise(counts)
        if base == "qidui":
            if is_qing:
                return 40, "清七对"
            if gui_count >= 1:
                return 20 * (2 ** (gui_count - 1)), "龙七对"
            return 10, "七对"
        if is_qing and base == "duidui":
            return 40, "清对"
        if is_qing and gui_count >= 1:
            return 20 * (2 ** (gui_count - 1)), "清闺"
        if base == "duidui" and gui_count >= 1:
            return 20 * (2 ** (gui_count - 1)), "闺对"
        if base == "duidui":
            return 10, "大对子"
        if is_qing:
            return 10, "清一色"
        if gui_count >= 1:
            return 5 * (2 ** (gui_count - 1)), "闺自摸"
        return 2, "平胡"

    def _qidui_decomposition(self, counts: List[int], ghost: int) -> Optional[Dict]:
        pairs = []
        ghost_left = ghost
        for tile in range(27):
            c = counts[tile]
            while c >= 2:
                pairs.append([tile, tile])
                c -= 2
            if c == 1:
                if ghost_left == 0:
                    return None
                pairs.append([tile, -1])
                ghost_left -= 1
        while ghost_left >= 2 and len(pairs) < 7:
            pairs.append([-1, -1])
            ghost_left -= 2
        if len(pairs) >= 7:
            pairs = pairs[:7]
            return {"type": "qidui", "melds": [self._format_pair(p) for p in pairs]}
        return None

    def _standard_decomposition(self, counts: List[int], ghost: int) -> Optional[Dict]:
        best = None
        for pair_used, pair_meld, counts_after in self._pair_candidates(counts, ghost):
            remaining_ghost = ghost - pair_used
            if remaining_ghost < 0:
                continue
            suit_solutions = []
            for base in (0, 9, 18):
                suit_counts = counts_after[base : base + 9]
                solutions = self._suit_solutions(tuple(suit_counts), remaining_ghost, base)
                suit_solutions.append(solutions)
            combined = [(0, [])]
            for solutions in suit_solutions:
                new_combined = []
                for g1, melds1 in combined:
                    for g2, melds2 in solutions:
                        new_combined.append((g1 + g2, melds1 + melds2))
                combined = new_combined
            for suit_used, melds in combined:
                ghost_left = remaining_ghost - suit_used
                if ghost_left < 0:
                    continue
                if ghost_left % 3 != 0:
                    continue
                extra_melds = ghost_left // 3
                total_melds = len(melds) + extra_melds
                if total_melds != 4:
                    continue
                ghost_melds = [["G", "G", "G"] for _ in range(extra_melds)]
                all_melds = [pair_meld] + melds + ghost_melds
                best = {"type": "duidui" if self._is_duidui_decomp(melds) else "standard", "melds": all_melds}
                return best
        return None

    def _pair_candidates(self, counts: List[int], ghost: int) -> List[Tuple[int, List[str], List[int]]]:
        candidates = []
        for tile in range(27):
            if counts[tile] >= 2:
                counts_after = counts[:]
                counts_after[tile] -= 2
                candidates.append((0, self._format_pair([tile, tile]), counts_after))
            elif counts[tile] == 1 and ghost >= 1:
                counts_after = counts[:]
                counts_after[tile] -= 1
                candidates.append((1, self._format_pair([tile, -1]), counts_after))
        if ghost >= 2:
            candidates.append((2, self._format_pair([-1, -1]), counts[:]))
        return candidates

    def _suit_solutions(self, counts: Tuple[int, ...], ghost: int, base: int) -> List[Tuple[int, List[List[str]]]]:
        memo: Dict[Tuple[Tuple[int, ...], int], List[Tuple[int, List[List[str]]]]] = {}

        def dfs(state: Tuple[int, ...], ghost_left: int) -> List[Tuple[int, List[List[str]]]]:
            key = (state, ghost_left)
            if key in memo:
                return memo[key]
            counts_list = list(state)
            i = 0
            while i < 9 and counts_list[i] == 0:
                i += 1
            if i >= 9:
                memo[key] = [(0, [])]
                return memo[key]
            results: List[Tuple[int, List[List[str]]]] = []
            count_i = counts_list[i]
            need_trip = max(0, 3 - count_i)
            if need_trip <= ghost_left:
                new_counts = counts_list[:]
                new_counts[i] = max(0, count_i - 3)
                meld = self._format_meld([base + i] * count_i + [-1] * need_trip)
                for g_used, melds in dfs(tuple(new_counts), ghost_left - need_trip):
                    results.append((g_used + need_trip, [meld] + melds))
            if i <= 6:
                need_seq = 0
                new_counts = counts_list[:]
                meld_tiles = []
                for offset in (0, 1, 2):
                    idx = i + offset
                    if new_counts[idx] > 0:
                        new_counts[idx] -= 1
                        meld_tiles.append(base + idx)
                    else:
                        need_seq += 1
                        meld_tiles.append(-1)
                if need_seq <= ghost_left:
                    meld = self._format_meld(meld_tiles)
                    for g_used, melds in dfs(tuple(new_counts), ghost_left - need_seq):
                        results.append((g_used + need_seq, [meld] + melds))
            memo[key] = results
            return results

        return dfs(counts, ghost)

    def _is_duidui_decomp(self, melds: List[List[str]]) -> bool:
        for meld in melds:
            if len(meld) != 3:
                continue
            if meld[0] != meld[1] or meld[1] != meld[2]:
                return False
        return True

    def _format_meld(self, tiles: List[int]) -> List[str]:
        return [tile_to_name(t) for t in tiles]

    def _format_pair(self, tiles: List[int]) -> List[str]:
        return [tile_to_name(t) for t in tiles]

    def _suits_used(self, counts: List[int]) -> int:
        suit_counts = [sum(counts[i * 9 : i * 9 + 9]) for i in range(3)]
        return sum(1 for c in suit_counts if c > 0)

    def _is_qingyise(self, counts: List[int]) -> bool:
        suit_counts = [sum(counts[i * 9 : i * 9 + 9]) for i in range(3)]
        return sum(1 for c in suit_counts if c > 0) == 1 and sum(suit_counts) > 0


class DecisionMaker:
    def __init__(self, scorer: LuzhouScorer) -> None:
        self.scorer = scorer

    def generate_recommendations(self, predictions: List[float], game_state: Dict) -> Tuple[List[Dict], str]:
        ranked = list(enumerate(predictions))
        ranked.sort(key=lambda x: x[1], reverse=True)
        min_multiplier = int(game_state.get("min_hu_multiplier", 1))
        hand = game_state.get("hand", [])
        mode = game_state.get("play_mode", "fast")
        filtered = []
        for tile, score in ranked:
            result = self.scorer.score(hand + [tile], game_state)
            fan = result["fan"]
            if fan >= min_multiplier:
                filtered.append(
                    {
                        "tile": tile,
                        "score": score,
                        "fan": fan,
                        "label": result["label"],
                        "melds": result["melds"],
                    }
                )
        warning = ""
        if not filtered:
            warning = f"未达到最小胡牌番数({min_multiplier}颗)"
            return [], warning
        if mode == "max":
            filtered.sort(key=lambda x: (x["fan"], x["score"]), reverse=True)
        else:
            filtered.sort(key=lambda x: x["score"], reverse=True)
        return filtered[:5], warning


class AdvancedRulePredictor:
    def __init__(self) -> None:
        self.rule_engine = RuleBasedRedZhongPredictor()
        self.context_analyzer = ContextAnalyzer()
        self.scorer = LuzhouScorer(self.rule_engine.calculator)
        self.decision_maker = DecisionMaker(self.scorer)

    def predict(self, game_state: Dict) -> Dict:
        context = self.context_analyzer.analyze(game_state)
        raw_predictions = self.rule_engine.predict(game_state)
        adjusted = self._apply_context_correction(raw_predictions, context)
        recommendations, warning = self.decision_maker.generate_recommendations(adjusted, game_state)
        action_advice = self._evaluate_actions(game_state)
        discard_suggestions = self._discard_suggestions(game_state)
        return {
            "predictions": adjusted,
            "recommendations": recommendations,
            "confidence": self._calculate_confidence(adjusted),
            "warning": warning,
            "action_advice": action_advice,
            "discard_suggestions": discard_suggestions,
        }

    def _apply_context_correction(self, predictions: List[float], context: Dict) -> List[float]:
        corrected = predictions[:]
        correction_factors = {
            "early_game": 1.2,
            "late_game": 0.8,
            "ahead": 1.1,
            "behind": 1.3,
            "defensive": 0.9,
            "offensive": 1.2,
        }
        for factor_name, factor_value in context["factors"].items():
            if factor_value > 0.5:
                correction = correction_factors.get(factor_name, 1.0)
                for i in range(TILE_COUNT):
                    corrected[i] *= correction
        total = sum(corrected)
        if total > 0:
            corrected = [c / total for c in corrected]
        return corrected

    def _calculate_confidence(self, predictions: List[float]) -> float:
        sorted_scores = sorted(predictions, reverse=True)
        if len(sorted_scores) >= 2:
            diff = sorted_scores[0] - sorted_scores[1]
            return min(diff * 5, 1.0)
        return 0.5

    def _evaluate_actions(self, game_state: Dict) -> Dict:
        hand = game_state.get("hand", [])
        current_discard = game_state.get("current_discard")
        current_discard_player = game_state.get("current_discard_player")
        current_player = game_state.get("current_player", 0)
        min_multiplier = int(game_state.get("min_hu_multiplier", 1))
        if current_discard is None:
            return {"action": "未知"}
        if current_discard_player == current_player:
            return {"action": "轮到自己出牌"}
        count = sum(1 for t in hand if t == current_discard)
        can_peng = count >= 2
        can_gang = count >= 3
        result = self.scorer.score(hand + [current_discard], game_state)
        can_hu = result["fan"] >= min_multiplier and result["fan"] > 0
        if can_hu:
            return {"action": "胡", "fan": result["fan"], "label": result["label"]}
        if can_gang:
            return {"action": "杠"}
        if can_peng:
            return {"action": "碰"}
        return {"action": "过"}

    def _discard_suggestions(self, game_state: Dict) -> List[Dict]:
        hand = game_state.get("hand", [])
        if not hand:
            return []
        candidates = []
        for i, tile in enumerate(hand):
            reduced_hand = hand[:i] + hand[i + 1 :]
            shanten = self.rule_engine.calculator.calculate(reduced_hand)
            candidates.append({"tile": tile, "shanten": shanten})
        candidates.sort(key=lambda x: (x["shanten"], x["tile"]))
        unique = []
        seen = set()
        for item in candidates:
            if item["tile"] in seen:
                continue
            seen.add(item["tile"])
            unique.append(item)
            if len(unique) >= 3:
                break
        return unique


predictor = AdvancedRulePredictor()


class BackendHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json({"status": "ok"})
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        if self.path != "/predict":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = self.rfile.read(length).decode("utf-8")
        try:
            game_state = json.loads(payload)
        except json.JSONDecodeError:
            self._send_json({"error": "invalid json"}, 400)
            return
        result = predictor.predict(game_state)
        self._send_json(result)

    def _send_json(self, data: Dict, code: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server(port: int) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", port), BackendHandler)
    server.serve_forever()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    run_server(args.port)
