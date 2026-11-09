import urllib.request
import soundata


DATA_HOME = "dataset/raw/urbansound8k"

dataset = soundata.initialize(
    "urbansound8k",
    data_home=DATA_HOME
)

print("Mulai download UrbanSound8K...")

try:
    dataset.download()

    print()
    print("======================================")
    print("DOWNLOAD SELESAI")
    print("======================================")
    print("Dataset berada di:")
    print(DATA_HOME)

except Exception as error:
    print()
    print("======================================")
    print("DOWNLOAD GAGAL")
    print("======================================")
    print(type(error).__name__)
    print(error)