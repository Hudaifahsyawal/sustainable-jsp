import numpy as np


def needleman_wunsch(seq1, seq2, match=0, mismatch=1, gap=1):
    n, m = len(seq1), len(seq2)

    score_matrix = np.zeros((n+1, m+1), dtype=int)

    for i in range(1, n+1):
        score_matrix[i][0] = score_matrix[i-1][0] + gap
    for j in range(1, m+1):
        score_matrix[0][j] = score_matrix[0][j-1] + gap

    for i in range(1, n+1):
        for j in range(1, m+1):
            match_score = match if seq1[i-1] == seq2[j-1] else mismatch
            score_matrix[i][j] = max(
                score_matrix[i-1][j-1] + match_score,
                score_matrix[i-1][j] + gap,
                score_matrix[i][j-1] + gap
            )

    align1, align2 = "", ""
    i, j = n, m
    while i > 0 or j > 0:
        current_score = score_matrix[i][j]

        if i > 0 and j > 0 and \
           current_score == score_matrix[i-1][j-1] + (match if seq1[i-1] == seq2[j-1] else mismatch):
            align1 = seq1[i-1] + align1
            align2 = seq2[j-1] + align2
            i -= 1
            j -= 1
        elif i > 0 and current_score == score_matrix[i-1][j] + gap:
            align1 = seq1[i-1] + align1
            align2 = "-" + align2
            i -= 1
        else:
            align1 = "-" + align1
            align2 = seq2[j-1] + align2
            j -= 1

    final_score = score_matrix[n][m]

    return align1, align2, score_matrix, final_score


def needleman_wunsch_dissimilarity(seq1, seq2, match_cost=0, mismatch_cost=-1, gap_cost=-1):
    """
    Global alignment ala Needleman-Wunsch untuk menghitung
    dissimilarity index = #mismatch + #gap minimum.

    seq1, seq2: list elemen (bisa tuple, string, dll)
    """
    n, m = len(seq1), len(seq2)

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + gap_cost
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + gap_cost

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                cost_diag = dp[i - 1][j - 1] + match_cost
            else:
                cost_diag = dp[i - 1][j - 1] + mismatch_cost

            cost_up = dp[i - 1][j] + gap_cost
            cost_left = dp[i][j - 1] + gap_cost

            dp[i][j] = min(cost_diag, cost_up, cost_left)

    final_cost = dp[n][m]
    return final_cost, dp


def scheduling_dissimilarity(solution1, solution2,
                             match_cost=0, mismatch_cost=1, gap_cost=1):
    """
    Menghitung dissimilarity index total antara dua solusi scheduling.
    Dihitung per mesin, lalu dijumlahkan.
    """
    all_machines = sorted(set(solution1.keys()) | set(solution2.keys()))
    per_machine_dissim = {}
    total_dissim = 0

    for m_id in all_machines:
        seq1 = solution1.get(m_id, [])
        seq2 = solution2.get(m_id, [])
        cost, _ = needleman_wunsch_dissimilarity(seq1, seq2, match_cost, mismatch_cost, gap_cost)
        per_machine_dissim[m_id] = cost
        total_dissim += cost

    return round(total_dissim, 3), per_machine_dissim
