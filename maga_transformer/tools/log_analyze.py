from typing import List
import re
import torch

log_file1 = "/data0/wangyin/workspace/FasterTransformer/tp1.log"
log_file2 = "/data0/wangyin/workspace/FasterTransformer/tp2.log"

class RowLog:
    def __init__(self, batch_idx: int, seq_idx: int, elements: List[float]) -> None:
        self.batch_idx = batch_idx
        self.seq_idx = seq_idx
        self.elements = elements

    def __str__(self) -> str:
        return f"batch_idx: {self.batch_idx}, seq_idx: {self.seq_idx}, elements: {self.elements}"

    def __repr__(self) -> str:
        return self.__str__()

class TensorLog:
    def __init__(self, name: str, seq_len: int, batch_size: int, dim: int, rows: List[RowLog] = []) -> None:
        self.name = name
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.dim = dim
        self.rows = rows

    def add_row(self, row: RowLog) -> None:
        self.rows.append(row)

def parse_log(file_path: str) -> List[TensorLog]:
    logs = []
    with open(file_path, "r") as f:
        lines = f.readlines().__iter__()

        try:
            next_line = lines.__next__()
            while (next_line):
                match = re.search(r'^(.+)\[(\d+)\s(\d+)\s(\d+)\]$', next_line)
                if match:
                    current_log = TensorLog(match.group(1), int(match.group(2)),
                                            int(match.group(3)), int(match.group(4)), [])
                    while True:
                        next_line = lines.__next__()
                        # checking the line only contains numbers and spaces
                        try:
                            elements = [float(x.replace('b_', '').replace('s_', '')) \
                                        for x in next_line.split()]
                            current_log.add_row(RowLog(int(elements[0]), int(elements[1]), elements[2:]))
                        except ValueError:
                            break
                    # print(f'loaded log {current_log.name} with {len(current_log.rows)} rows')
                    logs.append(current_log)
                else:
                    next_line = lines.__next__()
        except StopIteration:
            pass
        print(f'totally loaded {len(logs)} logs')

    return logs

def compare_logs(log1: List[TensorLog], log2: List[TensorLog]) -> None:
    for i in range(min(len(log1), len(log2))):
        if log1[i].name != log2[i].name:
            print(f'log name not match: {log1[i].name} vs {log2[i].name}')
            return
        if log1[i].seq_len != log2[i].seq_len:
            print(f'log seq_len not match: {log1[i].seq_len} vs {log2[i].seq_len}')
            return
        if log1[i].batch_size != log2[i].batch_size:
            print(f'log batch_size not match: {log1[i].batch_size} vs {log2[i].batch_size}')
            continue

        m1 = torch.tensor([row.elements for row in log1[i].rows])
        m2 = torch.tensor([row.elements for row in log2[i].rows])
        diff = torch.abs(m1 - m2)
        print(f'log [{log1[i].name}] diff: {diff}')

if __name__ == "__main__":
    logs1 = parse_log(log_file1)
    logs2 = parse_log(log_file2)
    compare_logs(logs1, logs2)
    print("done")


