import sys
from collections import deque


def solve():
    data = sys.stdin.buffer.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    m = int(data[idx]); idx += 1

    # Считываем артефакты: (ценность, номер_эпохи)
    items = []
    for epoch in range(n):
        for j in range(m):
            val = int(data[idx]); idx += 1
            items.append((val, epoch))

    # Сортируем по ценности
    items.sort()

    total = len(items)

    # Скользящее окно: ищем минимальный диапазон [items[l][0], items[r][0]],
    # содержащий хотя бы один артефакт каждой эпохи
    epoch_deques = [deque() for _ in range(n)]
    covered = 0          # сколько эпох покрыто
    current_min_sum = 0  # сумма минимумов по каждой эпохе в окне

    best_range = float('inf')
    best_sum = float('inf')
    best_l = 0
    best_r = 0

    l = 0
    for r in range(total):
        val_r, epoch_r = items[r]
        if not epoch_deques[epoch_r]:
            covered += 1
            current_min_sum += val_r
        epoch_deques[epoch_r].append(val_r)

        # Пытаемся сузить окно слева
        while covered == n:
            cur_range = items[r][0] - items[l][0]

            if cur_range < best_range or (cur_range == best_range and current_min_sum < best_sum):
                best_range = cur_range
                best_sum = current_min_sum
                best_l = l
                best_r = r

            val_l, epoch_l = items[l]
            epoch_deques[epoch_l].popleft()
            if not epoch_deques[epoch_l]:
                covered -= 1
                current_min_sum -= val_l
            else:
                # Минимум эпохи сменился: был val_l, стал новый front
                current_min_sum += epoch_deques[epoch_l][0] - val_l
            l += 1

    # Реконструкция: из оптимального окна берём минимальный артефакт каждой эпохи
    epoch_min = {}
    for i in range(best_l, best_r + 1):
        val, epoch = items[i]
        if epoch not in epoch_min:
            epoch_min[epoch] = val  # первый = минимальный (массив отсортирован)

    result = sorted(epoch_min.values())
    print(' '.join(map(str, result)))


solve()
