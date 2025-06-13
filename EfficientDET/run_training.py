import subprocess
import sys
import os
print(f"--> My Current Working Directory is: {os.getcwd()}")

def run_training():
    command = [
        "python",
        "train.py",
        "-c", "0",                 
        "-p", "CustomDET",           
        "--head_only", "False",        
        "--lr", "4e-3",  
        "--batch_size", "16",  
        "--load_weights", "weights/efficientdet-d0.pth",
        "--num_epochs", "10",
        # "--epochs", "10",    
        "--save_interval", "100",             
        "--num_workers", "8"
    ]


    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)

        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()

        process.wait()

        if process.returncode != 0:
            print(f"\n ERROR:  {process.returncode} ")
        else:
            print("\nDone")

    except FileNotFoundError:
        print("\nERROR")
    except Exception as e:
        print(f"\n{e}")

if __name__ == "__main__":
    run_training()
