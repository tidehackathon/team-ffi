import random
import subprocess
import xml.etree.ElementTree as ET

def get_free_gpu():
    print("Finner ledige GPU-er…")

    gpus = subprocess.run(["nvidia-smi", "-L"], capture_output=True).stdout
    gpus = gpus.decode().split("\n")
    gpus = [gpu for gpu in gpus if gpu.startswith("GPU")]
    gpus_free = []
    # print("gpus:", gpus)
    print(f"Det fins {len(gpus)} GPU-er. Sjekker nr. ", end="")
    for gpu_id in range(len(gpus)):
        print(gpu_id, end=", ")
        nvidia_out = subprocess.run(
            ["nvidia-smi", "-q", "-i", str(gpu_id), "-x"], capture_output=True).stdout
        nvidia_out = nvidia_out.decode()
    # print(nvidia_out)
        nvidia_tree = ET.fromstring(nvidia_out)
        processes = nvidia_tree.find("gpu").find(
            "processes").findall("process_info")
        n_processes = len(processes)
    # print("Number of processes:", n_processes)
        if n_processes == 0:
            gpus_free.append(gpu_id)
    print()
    print("Ubrukte GPU-er:", gpus_free)
    assert gpus_free, "Ingen ledige GPU-er."
    # print(nvidia_out)

    gpu_selected = random.choice(gpus_free)
    device_selected = f"cuda:{gpu_selected}"
    print(f"Valgte enhet {device_selected}.")
    return device_selected