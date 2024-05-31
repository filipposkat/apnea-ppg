import os
from pathlib import Path

def delete_redundant_batches(path):
    for root, dirs, files in os.walk(path):
        if "epoch-" in root:
            batches = []
            for currentFile in files:
                if "batch-" in currentFile and ".pt" in currentFile:
                    batch = currentFile.removesuffix(".pt")
                    batch = batch.removeprefix("batch-")
                    batches.append(int(batch))
            if len(batches) > 1:        
                batches.sort()        
                batches.pop()

                for b in batches:
                    file_name = f"batch-{b}.pt"
                    print("Deleting file: " + file_name)
                    os.remove(os.path.join(root, file_name))


if __name__ == "__main__":
    delete_redundant_batches(os.getcwd())