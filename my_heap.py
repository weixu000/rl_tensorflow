def heapify_up(heap, ind):
    parent = (ind - 1) // 2
    if parent >= 0 and heap[ind] < heap[parent]:
        heap[ind], heap[parent] = heap[parent], heap[ind]
        heapify_up(heap, parent)


def heapify_down(heap, ind):
    left, right = 2 * ind + 1, 2 * ind + 2
    if right < len(heap):
        if heap[left] < heap[right]:
            if heap[left] < heap[ind]:
                heap[ind], heap[left] = heap[left], heap[ind]
                heapify_down(heap, left)
        elif heap[right] < heap[ind]:
            heap[ind], heap[right] = heap[right], heap[ind]
            heapify_down(heap, right)
    elif left < len(heap) and heap[left] < heap[ind]:
        heap[ind], heap[left] = heap[left], heap[ind]
        heapify_down(heap, left)


def heapify_single(heap, ind):
    heapify_up(heap, ind)
    heapify_down(heap, ind)
    pass


def check_heap_invariant(heap):
    for parent in range(len(heap) // 2):
        for child in 2 * parent + 1, 2 * parent + 2:
            if child < len(heap) and heap[child] < heap[parent]:
                print('Heap invariant broken: ', parent, child)
                break
